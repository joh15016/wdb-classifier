"""Bounded Kowalski find queries and resumable single-band ZTF photometry."""

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
from .contracts import MIN_PERIOD

BANDS = {1: "g", 2: "r", 3: "i"}
COLUMNS = ("source_id", "band", "time", "mag", "mag_err", "ra", "dec", "time_system")


@dataclass(frozen=True)
class ZTFConfig:
    host: str
    collection: str
    port: int = 443
    timeout: int = 30
    batch_size: int = 10
    bands: tuple[str, ...] = ("g", "r", "i")
    program_ids: tuple[int, ...] = (1, 2, 3)
    min_cadence_days: float = MIN_PERIOD
    max_hjd: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "program_ids", tuple(self.program_ids))
        if not re.fullmatch(r"[a-zA-Z0-9.-]+", self.host) or not re.fullmatch(
            r"ZTF_sources_\d{8}", self.collection
        ):
            raise ValueError("Specify a Kowalski hostname and explicit ZTF_sources_YYYYMMDD collection")
        if not (1 <= self.batch_size <= 100 and 1 <= self.timeout <= 120 and 1 <= self.port <= 65535):
            raise ValueError("Use batches of 1–100 IDs, timeout 1–120 seconds, and a valid port")
        if not self.bands or set(self.bands) - set(BANDS.values()) or set(self.program_ids) - {1, 2, 3}:
            raise ValueError("Invalid ZTF bands or program IDs")
        if not np.isfinite(self.min_cadence_days) or self.min_cadence_days < 0:
            raise ValueError("min_cadence_days must be finite and nonnegative")
        if self.max_hjd is not None and (not np.isfinite(self.max_hjd) or self.max_hjd <= 0):
            raise ValueError("max_hjd must be positive and finite")

    @classmethod
    def load(cls, path):
        return cls(**tomllib.loads(Path(path).read_text())["ztf"])


def source_ids(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = {"ztf_id", "source_id"}.intersection(frame)
    if len(names) != 1:
        raise ValueError("The CSV must have one ID column named ztf_id or source_id")
    ids = frame[names.pop()].tolist()
    if not ids or any(not re.fullmatch(r"[1-9][0-9]{0,18}", sid) or int(sid) >= 2**63 for sid in ids):
        raise ValueError("Use positive integer ZTF source IDs, not alert names or floating-point IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ZTF IDs in the request")
    return ids


def connect(config, token_env):
    token = os.environ.get(token_env)
    if not token:
        raise ValueError(f"Set {token_env} locally before fetching; no token belongs in the TOML file")
    from penquins import Kowalski

    try:
        return Kowalski(
            timeout=config.timeout,
            verbose=False,
            instances={
                "kowalski": {
                    "protocol": "https",
                    "host": config.host,
                    "port": config.port,
                    "token": token,
                }
            },
        )
    except Exception:
        raise RuntimeError(
            "Could not initialize the Kowalski client; check credentials and connectivity"
        ) from None


def build_query(ids, config):
    epochs = {"$gt": 0}
    if config.max_hjd is not None:
        epochs["$lte"] = config.max_hjd
    filters = {"_id": {"$in": [int(sid) for sid in ids]}, "data.hjd": epochs}
    if config.program_ids:
        filters["data.programid"] = {"$in": list(config.program_ids)}
    return {
        "query_type": "find",
        "query": {
            "catalog": config.collection,
            "filter": filters,
            "projection": dict.fromkeys(
                (
                    "_id",
                    "filter",
                    "ra",
                    "dec",
                    "data.hjd",
                    "data.mag",
                    "data.magerr",
                    "data.catflags",
                    "data.programid",
                ),
                1,
            ),
        },
    }


def query_batch(client, ids, config, *, attempts=3):
    """Never interpret a failed/malformed response as a successful empty search."""
    query = build_query(ids, config)
    for attempt in range(attempts):
        try:
            replies = client.query(queries=[query], use_batch_query=True, max_n_threads=1)
            responses = replies["kowalski"]
            if (
                not isinstance(responses, list)
                or len(responses) != 1
                or responses[0].get("status") != "success"
                or not isinstance(responses[0].get("data"), list)
            ):
                raise ValueError("Unexpected Kowalski response")
            documents = responses[0]["data"]
            by_id = {str(doc["_id"]): doc for doc in documents}
            if len(by_id) != len(documents) or set(by_id) - set(ids):
                raise ValueError("Unexpected or repeated returned ID")
            return by_id
        except Exception:
            if attempt + 1 == attempts:
                # Do not echo the client's request, error body, or credentials.
                raise RuntimeError(
                    "Kowalski query failed; check access, collection and connection. Batch not cached."
                ) from None
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


def clean_document(doc, config):
    band = BANDS.get(doc.get("filter"))
    if band not in config.bands:
        return []
    epochs = []
    for point in doc.get("data", []):
        program = point.get("programid")
        if point.get("catflags", 1) != 0 or (
            config.program_ids and program not in (*config.program_ids, None)
        ):
            continue
        try:
            t, m, e = (float(point[k]) for k in ("hjd", "mag", "magerr"))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (t, m, e)) or t <= 0 or e <= 0:
            continue
        if config.max_hjd is None or t <= config.max_hjd:
            epochs.append((t, m, e))
    kept, last = [], -np.inf
    for t, m, e in sorted(epochs, key=lambda p: p[0]):
        if t - last >= config.min_cadence_days:
            kept.append(
                dict(
                    source_id=str(doc["_id"]),
                    band=band,
                    time=t,
                    mag=m,
                    mag_err=e,
                    ra=doc.get("ra"),
                    dec=doc.get("dec"),
                    time_system="HJD_as_returned",
                )
            )
            last = t
    return kept


def write_parquet(path, frame):
    path = Path(path)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def fetch(ids_path, config, output, *, max_new_sources=20, token_env="WDB_ZTF_TOKEN", client=None):
    if max_new_sources < 0:
        raise ValueError("max_new_sources cannot be negative")
    ids, output = source_ids(ids_path), Path(output)
    with run_lock(output):
        guard(
            output / "manifest.json",
            json.loads(
                json.dumps(
                    dict(
                        input_sha256=digest(ids_path),
                        config=asdict(config),
                        implementation=digest(__file__),
                    )
                )
            ),
        )
        directory = output / "sources"
        directory.mkdir(exist_ok=True)
        pending = [sid for sid in ids if not (directory / f"{sid}.json").exists()][:max_new_sources]
        if pending and client is None:
            client = connect(config, token_env)
        for i in range(0, len(pending), config.batch_size):
            batch = pending[i : i + config.batch_size]
            documents = query_batch(client, batch, config)
            for sid in batch:
                rows = clean_document(documents[sid], config) if sid in documents else []
                state = "ok" if rows else ("no_clean_epochs" if sid in documents else "no_matching_document")
                write_json(directory / f"{sid}.json", dict(source_id=sid, status=state, observations=rows))
        records = [
            json.loads((directory / f"{sid}.json").read_text())
            for sid in ids
            if (directory / f"{sid}.json").exists()
        ]
        photometry = pd.DataFrame([row for item in records for row in item["observations"]], columns=COLUMNS)
        audit = pd.DataFrame(
            [
                dict(source_id=r["source_id"], status=r["status"], n_obs=len(r["observations"]))
                for r in records
            ],
            columns=["source_id", "status", "n_obs"],
        )
        write_parquet(output / "photometry.parquet", photometry)
        write_parquet(output / "retrieval_audit.parquet", audit)
        status = dict(
            requested=len(ids),
            completed=len(records),
            new_sources=len(pending),
            sources_with_photometry=int(audit.status.eq("ok").sum()),
            complete=len(records) == len(ids),
        )
        write_json(output / "status.json", status)
        return status
