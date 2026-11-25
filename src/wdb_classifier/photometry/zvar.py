"""Authenticated, resumable ZVAR FPW candidates; no photometry or label queries.

External candidates are hypotheses, not a photometry-hash-compatible local
periodogram cache. Their band, bin, match and significance provenance stay
separate from local LS/CE/AoV results.
"""

import json
import os
import re
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import digest, guard, run_lock, write_json
from ..data import read_table
from .contracts import light_curves
from .ztf import write_parquet

SCHEMA = "wdb_zvar_fpw_cache_v1"
PERIOD_FIELDS = (
    "band",
    "field",
    "ccd",
    "quad",
    "period",
    "significance",
    "bin",
    "duplicate_freq",
    "fap",
    "n_sigma",
)
FPW_FEATURES = (
    "zvar_fpw_available",
    "zvar_fpw_period",
    "zvar_fpw_significance",
    "zvar_fpw_fap",
    "zvar_fpw_n_sigma",
    "zvar_fpw_bin",
    "zvar_fpw_candidate_count",
)


@dataclass(frozen=True)
class ZVARConfig:
    base_url: str = "https://api.zvar.caltech.edu/imdiff"
    radius_arcsec: float = 3.0
    accept_radius_arcsec: float = 1.5
    batch_size: int = 10
    timeout: int = 30

    def __post_init__(self):
        # Never forward Basic Auth to an arbitrary URL or through redirects.
        if self.base_url not in ("https://api.zvar.caltech.edu/imdiff", "https://hypernova.caltech.edu/api"):
            raise ValueError("Use a documented HTTPS ZVAR endpoint")
        if not 0 < self.accept_radius_arcsec <= self.radius_arcsec <= 120:
            raise ValueError("Require 0 < accepted radius <= cone radius <= 120 arcsec")
        if not 1 <= self.batch_size <= 100 or not 1 <= self.timeout <= 120:
            raise ValueError("Use batches of 1–100 and timeouts of 1–120 seconds")

    @classmethod
    def load(cls, path):
        return cls(**tomllib.loads(Path(path).read_text())["zvar"])


def separation_arcsec(ra, dec, other_ra, other_dec):
    first, second = np.deg2rad([dec, other_dec])
    delta = np.deg2rad(other_ra - ra)
    h = np.sin((second - first) / 2) ** 2 + np.cos(first) * np.cos(second) * np.sin(delta / 2) ** 2
    return float(np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600)


def validate_ids(ids):
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", sid) is None for sid in ids):
        raise ValueError("Source IDs must be filename-safe alphanumeric strings")


def positions(table):
    """Only stable source coordinates are read; never labels or known periods."""
    curves = light_curves(table)
    groups = dict(tuple(table.assign(source_id=table.source_id.astype(str)).groupby("source_id", sort=False)))
    result = {}
    for curve in curves:
        part = groups[curve.source_id]
        location = None
        if {"ra", "dec"}.issubset(part):
            coords = part[["ra", "dec"]].to_numpy(float)
            valid = (
                np.isfinite(coords).all(axis=1)
                & (coords[:, 0] >= 0)
                & (coords[:, 0] < 360)
                & (np.abs(coords[:, 1]) <= 90)
            )
            if valid.any():
                ra, dec = coords[valid][0]
                if any(separation_arcsec(ra, dec, a, d) > 0.1 for a, d in coords[valid]):
                    raise ValueError(f"Inconsistent coordinates for source {curve.source_id}")
                location = dict(ra=float(ra), dec=float(dec))
        result[curve.source_id] = location
    return result


def normalize_matches(objects, location, config):
    """Do not silently choose the closest of several distinct PS1 stars."""
    matches = {}
    for obj in objects:
        try:
            psid = obj["psid"]
            ra, dec = float(obj["ra"]), float(obj["dec"])
            periods = obj.get("periods") or []
            if (
                not isinstance(psid, str)
                or not psid.isdecimal()
                or not np.isfinite([ra, dec]).all()
                or not 0 <= ra < 360
                or not -90 <= dec <= 90
                or not isinstance(periods, list)
            ):
                raise ValueError
            offset = separation_arcsec(location["ra"], location["dec"], ra, dec)
            if offset > config.radius_arcsec + 1e-5 or psid in matches:
                raise ValueError
            rows = []
            for row in periods:
                if (
                    not isinstance(row, dict)
                    or row.get("band") not in ("g", "r")
                    or not np.isfinite(float(row["period"]))
                    or float(row["period"]) <= 0
                    or not np.isfinite(float(row["significance"]))
                    or row.get("bin") not in (5, 10, 20)
                    or not isinstance(row.get("duplicate_freq", False), bool)
                ):
                    raise ValueError
                values = {key: row.get(key) for key in PERIOD_FIELDS}
                values["period"] = float(row["period"])
                values["significance"] = float(row["significance"])
                values["duplicate_freq"] = row.get("duplicate_freq", False)
                for key in ("fap", "n_sigma"):
                    if values[key] is not None:
                        number = float(values[key])
                        values[key] = number if np.isfinite(number) else None
                if values["fap"] is not None and not 0 <= float(values["fap"]) <= 1:
                    raise ValueError
                rows.append(values)
            matches[psid] = dict(psid=psid, ra=ra, dec=dec, separation_arcsec=offset, periods=rows)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("Malformed ZVAR object/period response; batch not cached") from None
    values = list(matches.values())
    if not values:
        state = "no_match"
    elif len(values) != 1:
        state = "ambiguous_match"
    elif values[0]["separation_arcsec"] > config.accept_radius_arcsec:
        state = "outside_accept_radius"
    else:
        state = "matched"
    return state, values


def query_batch(session, batch, config):
    payload = {"positions": [dict(name=sid, radius=config.radius_arcsec, **loc) for sid, loc in batch]}
    try:
        response = session.post(
            config.base_url + "/cone_search", json=payload, timeout=(5, config.timeout), allow_redirects=False
        )
    except Exception:
        raise RuntimeError("ZVAR query failed; check access and connectivity. Batch not cached.") from None
    if response.status_code != 200:
        raise RuntimeError(f"ZVAR query HTTP {response.status_code}; batch not cached")
    try:
        data = response.json()
    except Exception:
        raise RuntimeError("ZVAR returned invalid JSON; batch not cached") from None
    if not isinstance(data, dict) or any(not isinstance(data.get(sid), list) for sid, _ in batch):
        raise ValueError("Malformed ZVAR cone-search response; batch not cached")
    # Validate the whole batch before writing any of its source checkpoints.
    return {sid: normalize_matches(data[sid], loc, config) for sid, loc in batch}


def fetch(photometry, config, output, *, max_new_sources=20, session=None):
    """Cache successful cone queries. Failed requests are never cached as no match."""
    if max_new_sources < 0:
        raise ValueError("max_new_sources cannot be negative")
    locations = positions(read_table(photometry))
    validate_ids(locations)
    output = Path(output)
    if Path(photometry).resolve() in (
        (output / "period_candidates.parquet").resolve(),
        (output / "match_audit.parquet").resolve(),
    ):
        raise ValueError("Choose a ZVAR output directory separate from input photometry")
    with run_lock(output):
        guard(
            output / "manifest.json",
            dict(
                schema=SCHEMA,
                input_sha256=digest(photometry),
                config=asdict(config),
                implementation=digest(__file__),
            ),
        )
        directory = output / "sources"
        directory.mkdir(exist_ok=True)
        pending = [(sid, loc) for sid, loc in locations.items() if not (directory / f"{sid}.json").exists()][
            :max_new_sources
        ]
        owned = None
        network = [(sid, loc) for sid, loc in pending if loc is not None]
        if network and session is None:
            import requests

            username, password = os.environ.get("ZVAR_USERNAME"), os.environ.get("ZVAR_PASSWORD")
            if not username or not password:
                raise ValueError(
                    "Set ZVAR_USERNAME and ZVAR_PASSWORD locally; no credentials belong in config"
                )
            owned = requests.Session()
            owned.trust_env = False
            owned.auth = (username, password)
            session = owned
        started = time.monotonic()
        try:
            for sid, loc in pending:
                if loc is None:
                    write_json(
                        directory / f"{sid}.json",
                        dict(source_id=sid, query_position=None, status="missing_coordinates", matches=[]),
                    )
            for offset in range(0, len(network), config.batch_size):
                batch = network[offset : offset + config.batch_size]
                results = query_batch(session, batch, config)
                for sid, loc in batch:
                    state, matches = results[sid]
                    write_json(
                        directory / f"{sid}.json",
                        dict(source_id=sid, query_position=loc, status=state, matches=matches),
                    )
        finally:
            if owned is not None:
                owned.close()
        records = [
            json.loads((directory / f"{sid}.json").read_text())
            for sid in locations
            if (directory / f"{sid}.json").exists()
        ]
        audit = pd.DataFrame(
            [
                dict(source_id=r["source_id"], status=r["status"], n_matches=len(r["matches"]))
                for r in records
            ],
            columns=["source_id", "status", "n_matches"],
        )
        rows = [
            dict(
                source_id=r["source_id"],
                psid=match["psid"],
                separation_arcsec=match["separation_arcsec"],
                **p,
            )
            for r in records
            for match in r["matches"]
            for p in match["periods"]
        ]
        write_parquet(output / "match_audit.parquet", audit)
        write_parquet(
            output / "period_candidates.parquet",
            pd.DataFrame(rows, columns=["source_id", "psid", "separation_arcsec", *PERIOD_FIELDS]),
        )
        status = dict(
            requested=len(locations),
            completed=len(records),
            new_sources=len(pending),
            matched=int(audit.status.eq("matched").sum()),
            complete=len(records) == len(locations),
            seconds=time.monotonic() - started,
        )
        write_json(output / "status.json", status)
        return status


def load_candidates(directory, table, config, *, band_policy="any_gr"):
    """Read immutable, separately fingerprinted external hypotheses, never local peaks."""
    if band_policy not in ("any_gr", "same_band"):
        raise ValueError("Choose any_gr or same_band for ZVAR bands")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Not a supported ZVAR FPW cache")
    match_config = ZVARConfig(**manifest["config"])
    curves = light_curves(table)
    locations = positions(table)
    validate_ids(locations)
    selected, fingerprints = {}, {"manifest": digest(directory / "manifest.json")}
    for curve in curves:
        sid = curve.source_id
        path = directory / "sources" / f"{sid}.json"
        if not path.exists():
            fingerprints[sid] = None
            selected[sid] = dict(status="not_cached", candidates=[])
            continue
        fingerprints[sid] = digest(path)
        record = json.loads(path.read_text())
        previous, current = record.get("query_position"), locations[sid]
        same_location = (
            previous == current
            if previous is None or current is None
            else (separation_arcsec(previous["ra"], previous["dec"], current["ra"], current["dec"]) <= 0.1)
        )
        if record["source_id"] != sid or not same_location:
            raise ValueError(f"ZVAR cache coordinates do not correspond to source {sid}")
        state = record["status"]
        matches = record["matches"]
        if current is not None:
            actual_state, matches = normalize_matches(matches, previous, match_config)
            if actual_state != state:
                raise ValueError("ZVAR checkpoint match status does not match its objects")
        elif state != "missing_coordinates" or matches:
            raise ValueError("Invalid coordinate-free ZVAR checkpoint")
        rows = []
        if state == "matched":
            if len(matches) != 1:
                raise ValueError("Invalid matched ZVAR checkpoint")
            for row in matches[0]["periods"]:
                period = row["period"]
                if (
                    config.min_period_days <= period <= config.max_period_days
                    and not row.get("duplicate_freq", False)
                    and (band_policy == "any_gr" or row["band"] == curve.band)
                ):
                    rows.append(dict(psid=matches[0]["psid"], **row))
            # Calibrated FAP is comparable across bin choices, unlike raw z.
            # If it is absent, deterministically prefer the finest bin, then
            # significance within that bin. Never apply a WDB/significance veto.
            rows.sort(
                key=lambda r: (
                    r.get("fap") is None,
                    r["fap"] if r.get("fap") is not None else 1,
                    -r["bin"],
                    -r["significance"],
                )
            )
            if not rows:
                state = "no_usable_periods"
        selected[sid] = dict(status=state, candidates=rows)
    return selected, dict(
        files=fingerprints,
        band_policy=band_policy,
        exclude_duplicate_frequencies=True,
        ranking="calibrated_fap_then_bin_then_significance",
    )


def fpw_features(candidates):
    values = dict.fromkeys(FPW_FEATURES, np.nan)
    values.update(zvar_fpw_available=float(bool(candidates)), zvar_fpw_candidate_count=float(len(candidates)))
    if candidates:
        row = candidates[0]
        values.update(
            zvar_fpw_period=row["period"],
            zvar_fpw_significance=row["significance"],
            zvar_fpw_fap=row.get("fap"),
            zvar_fpw_n_sigma=row.get("n_sigma"),
            zvar_fpw_bin=row["bin"],
        )
    return values
