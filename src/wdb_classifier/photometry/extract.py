"""Checkpointed period/Fourier/eclipses from local or retrieved photometry."""

import hashlib
import importlib.metadata
import json
import math
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import digest, guard, run_lock, write_json
from ..data import read_table
from . import eclipse
from .contracts import PeriodConfig, light_curves
from .period import PeriodExtractor
from .ztf import write_parquet
from .zvar import FPW_FEATURES, fpw_features, load_candidates

FOURIER = tuple(PeriodExtractor._unpack_fourier(np.zeros(14)))
FEATURES = ("period", "period_significance", *FOURIER, *eclipse.SELECTED_FIELDS, *eclipse.MULTI_FIELDS)


def curve_digest(curve):
    # Hash the actual ordered float64 photometry supplied to the search.
    array = np.asarray([curve.time, curve.mag, curve.mag_err], dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def peak_columns(config):
    return [
        f"{key}_{k}_{algo}"
        for algo in config.algorithms
        for key in ("period", "significance")
        for k in range(config.top_n_periods)
    ]


class CachedExtractor(PeriodExtractor):
    """Reuse the ordinary Fourier/consensus path with no periodogram evaluation."""

    def __init__(self, config, row):
        super().__init__(config)
        self.row = row
        # The cached path supplies one source at a time and never uses a trial grid.
        self._grid = np.empty(0, dtype=np.float32)

    @property
    def algos(self):
        return dict.fromkeys(self._cfg.algorithms)

    def _run_algorithm(self, algo_name, algo, times, mags, errs, n_top):
        if len(times) != 1:
            raise ValueError("Cached peak replay accepts exactly one curve")
        return tuple(
            np.array([[self.row[f"{key}_{k}_{algo_name}"] for k in range(n_top)]], float)
            for key in ("period", "significance")
        )


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def preflight_grid(config, baseline, max_trials):
    if not np.isfinite(baseline) or baseline <= 0 or max_trials <= 0:
        raise ValueError("A positive baseline and trial limit are required")
    low = max(2 / baseline, 1 / config.max_period_days)
    high = 1 / config.min_period_days
    count = math.ceil((high - low) * config.samples_per_peak * baseline)
    if count <= 0:
        raise ValueError("No trial periods fit the supplied baseline")
    if count > max_trials:
        raise ValueError(
            f"Grid needs approximately {count:,} trials per source/algorithm; limit is {max_trials:,}"
        )
    return count


def compute(curve, extractor, min_obs, external_periods=()):
    if curve.n_obs < min_obs or np.ptp(curve.time) <= 0:
        return {}, {}, "insufficient_photometry"
    result = extractor.extract([[curve]])[0]
    top = result["period_top"]
    if set(top) != set(extractor._cfg.algorithms):
        raise RuntimeError("A requested period algorithm failed; source was not checkpointed")
    peaks = {
        f"{key}_{k}_{algo}": value
        for key, arrays in (("period", top), ("significance", result["significance_top"]))
        for algo, values in arrays.items()
        for k, value in enumerate(values)
    }
    values = {name: result[name] for name in ("period", "period_significance", *FOURIER)}
    if not np.isfinite(values["period"]):
        return values, peaks, "no_period"
    hypotheses = {**top, "ZVAR_FPW": list(external_periods)} if external_periods else top
    eclipse_values, _, _ = eclipse.extract(curve.time, curve.mag, curve.mag_err, values["period"], hypotheses)
    values.update(eclipse_values)
    return values, peaks, "ok"


def compute_external(curve, candidates):
    """Fit external FPW hypotheses without running or faking local periodograms."""
    selected = float(candidates[0]["period"])
    t, m, e, _, _ = PeriodExtractor._prepare(curve)
    raw = PeriodExtractor._fourier([t], [m], [e], np.array([selected]))
    if raw is None:
        raise RuntimeError("Fourier calculation failed; source was not checkpointed")
    values = dict(period=selected, period_significance=np.nan, **PeriodExtractor._unpack_fourier(raw[0]))
    fitted, _, _ = eclipse.extract(
        curve.time, curve.mag, curve.mag_err, selected, {"ZVAR_FPW": [row["period"] for row in candidates]}
    )
    values.update(fitted)
    return values, {}, "ok"


def run(
    photometry,
    config_path,
    output,
    *,
    cached_peaks=None,
    zvar_cache=None,
    zvar_mode="supplement",
    zvar_bands="any_gr",
    max_new_sources=1,
    seconds=120,
    baseline_days=None,
    max_trials=7_000_000,
    threads=1,
):
    """Run one source at a time. Time budget is checked between sources."""
    import periodfind

    if max_new_sources < 0 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Use a nonnegative source limit and positive time budget")
    if zvar_mode not in ("supplement", "prefer") or (zvar_cache is None and zvar_mode != "supplement"):
        raise ValueError("Use supplement or prefer with an explicit --zvar-cache")
    document = tomllib.loads(Path(config_path).read_text())
    config = PeriodConfig(**document["period"])
    min_obs = document.get("extraction", {}).get("min_observations", 50)
    if type(min_obs) is not int or min_obs < 10:
        raise ValueError("Eclipse features require at least ten observations")
    if not hasattr(periodfind, "FourierDecomposition") or not hasattr(periodfind, "set_device"):
        raise ImportError("Use the CPU-capable periodfind build documented in PHOTOMETRY.md")
    periodfind.set_device("cpu")
    table = read_table(photometry)
    curves = light_curves(table)
    external, external_manifest = {}, None
    if zvar_cache is not None:
        if zvar_mode == "supplement" and "FPW" in config.algorithms:
            raise ValueError(
                "For supplementation, remove FPW from local algorithms; ZVAR supplies it separately"
            )
        external, external_manifest = load_candidates(zvar_cache, table, config, band_policy=zvar_bands)
    output = Path(output)
    if zvar_cache is not None and Path(zvar_cache).resolve() == output.resolve():
        raise ValueError("Keep external ZVAR caches separate from extracted features")
    if any(
        Path(p).resolve() in {(output / "features.parquet").resolve(), (output / "peaks.parquet").resolve()}
        for p in (photometry, cached_peaks)
        if p is not None
    ):
        raise ValueError("Choose an output directory separate from input tables")

    def external_only(curve):
        return zvar_mode == "prefer" and bool(external.get(curve.source_id, {}).get("candidates"))

    eligible = [c for c in curves if c.n_obs >= min_obs and np.ptp(c.time) > 0]
    local_curves = [c for c in eligible if not external_only(c)]
    cache, fresh = None, None
    baseline, grid_count = None, None
    if cached_peaks is not None:
        cache = read_table(cached_peaks)
        required = {"source_id", "band", "curve_sha256", *peak_columns(config)}
        if not required.issubset(cache) or cache.duplicated(["source_id", "band"]).any():
            raise ValueError(
                "Cached peaks require unique source/band keys, curve hashes and all requested peaks"
            )
        cache = cache.assign(source_id=cache.source_id.astype(str)).set_index(["source_id", "band"])
        for curve in local_curves:
            key = (curve.source_id, curve.band)
            if key not in cache.index or cache.loc[key, "curve_sha256"] != curve_digest(curve):
                raise ValueError(f"Cached peaks do not correspond to the supplied photometry: {key}")
    else:
        observed = float(max((np.ptp(c.time) for c in eligible), default=0))
        baseline = observed if baseline_days is None else float(baseline_days)
        if baseline < observed:
            raise ValueError("The fixed grid baseline cannot be shorter than a searched light curve")
        if local_curves:
            grid_count = preflight_grid(config, baseline, max_trials)
    implementation = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    settings = clean_json(
        dict(
            photometry_sha256=digest(photometry),
            config=asdict(config),
            min_observations=min_obs,
            peak_input_sha256=None if cached_peaks is None else digest(cached_peaks),
            baseline_days=baseline,
            grid_count=grid_count,
            backend="cpu",
            threads=threads,
            versions={name: importlib.metadata.version(name) for name in ("numpy", "astropy", "periodfind")},
            periodfind_entrypoint_sha256=digest(periodfind.__file__),
            implementation=implementation,
            band_policy="most valid epochs per source ID; first input band breaks ties",
            zvar=None if external_manifest is None else dict(mode=zvar_mode, **external_manifest),
        )
    )
    started, new = time.monotonic(), 0
    with run_lock(output):
        guard(output / "manifest.json", settings)
        directory = output / "sources"
        directory.mkdir(exist_ok=True)
        for index, curve in enumerate(curves):
            path = directory / f"{index:08d}.json"
            if path.exists():
                continue
            if new >= max_new_sources or time.monotonic() - started >= seconds:
                break
            tic = time.monotonic()
            provider = "not_evaluated"
            candidates = external.get(curve.source_id, {}).get("candidates", [])
            if curve.n_obs < min_obs or np.ptp(curve.time) <= 0:
                values, peaks, state = {}, {}, "insufficient_photometry"
            elif external_only(curve):
                provider = "zvar_fpw"
                values, peaks, state = compute_external(curve, candidates)
            else:
                if cache is None:
                    if fresh is None:
                        fresh = PeriodExtractor(config)
                        fresh._baseline = baseline
                        fresh._grid = fresh._build_freq_grid(baseline)
                        if len(fresh._grid) != grid_count:
                            raise RuntimeError("Trial-grid preflight and actual size disagree")
                    extractor, provider = fresh, "local_search"
                else:
                    extractor = CachedExtractor(config, cache.loc[(curve.source_id, curve.band), :])
                    provider = "cached_local_peaks"
                values, peaks, state = compute(
                    curve, extractor, min_obs, [row["period"] for row in candidates]
                )
            provenance = {}
            if zvar_cache is not None:
                values.update(fpw_features(candidates))
                provenance = dict(
                    period_provider=provider,
                    zvar_status=external[curve.source_id]["status"],
                    zvar_candidates=candidates,
                )
            write_json(
                path,
                clean_json(
                    dict(
                        source_id=curve.source_id,
                        band=curve.band,
                        n_obs=curve.n_obs,
                        curve_sha256=curve_digest(curve),
                        status=state,
                        features=values,
                        peaks=peaks,
                        seconds=time.monotonic() - tic,
                        **provenance,
                    )
                ),
            )
            new += 1
            if new == 1 or new % 25 == 0:
                print(f"Processed {new} new sources; latest {curve.source_id}: {state}", flush=True)
        records = []
        for index, curve in enumerate(curves):
            path = directory / f"{index:08d}.json"
            if path.exists():
                item = json.loads(path.read_text())
                if item["source_id"] != curve.source_id or item["curve_sha256"] != curve_digest(curve):
                    raise ValueError("A checkpoint does not match its photometry")
                records.append(item)
        identifiers = ["source_id", "band", "n_obs", "curve_sha256", "status"]
        features = FEATURES
        if zvar_cache is not None:
            identifiers.extend(["period_provider", "zvar_status"])
            features = (*FEATURES, *FPW_FEATURES)
            zvar_rows = [
                dict(source_id=r["source_id"], **row) for r in records for row in r["zvar_candidates"]
            ]
            from .zvar import PERIOD_FIELDS

            write_parquet(
                output / "zvar_candidates.parquet",
                pd.DataFrame(zvar_rows, columns=["source_id", "psid", *PERIOD_FIELDS]),
            )
        for kind, columns in (("features", features), ("peaks", peak_columns(config))):
            # External FPW candidates never masquerade as local peak-cache rows.
            included = records if kind == "features" else [r for r in records if r["peaks"]]
            table = pd.DataFrame(
                [{**{k: r[k] for k in identifiers}, **r[kind]} for r in included],
                columns=[*identifiers, *columns],
            )
            for name in columns:
                table[name] = pd.to_numeric(table[name]).astype(float)
            write_parquet(output / f"{kind}.parquet", table)
        schema = dict(name="wdb_period_eclipse50_v1", columns=list(features))
        if zvar_cache is not None:
            schema.update(
                name=f"wdb_period_eclipse_zvar_{zvar_mode}57_v1",
                period_significance="local-search score only; NaN for external FPW-selected periods",
                zvar_significance="raw FPW score, FAP and n_sigma are separately namespaced",
                zvar_mode=zvar_mode,
                zvar_band_policy=zvar_bands,
            )
        write_json(output / "schema.json", schema)
        status = dict(
            requested=len(curves),
            processed=len(records),
            new_sources=new,
            successful=sum(r["status"] == "ok" for r in records),
            complete=len(records) == len(curves),
            cached_peak_replay=cache is not None,
            grid_count=grid_count,
            seconds=time.monotonic() - started,
        )
        if zvar_cache is not None:
            status.update(
                zvar_mode=zvar_mode,
                sources_with_zvar_candidates=sum(bool(r["zvar_candidates"]) for r in records),
                external_only_sources=sum(r["period_provider"] == "zvar_fpw" for r in records),
            )
        write_json(output / "status.json", status)
        return status
