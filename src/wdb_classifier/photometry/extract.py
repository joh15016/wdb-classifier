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


def compute(curve, extractor, min_obs):
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
    eclipse_values, _, _ = eclipse.extract(curve.time, curve.mag, curve.mag_err, values["period"], top)
    values.update(eclipse_values)
    return values, peaks, "ok"


def run(
    photometry,
    config_path,
    output,
    *,
    cached_peaks=None,
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
    document = tomllib.loads(Path(config_path).read_text())
    config = PeriodConfig(**document["period"])
    min_obs = document.get("extraction", {}).get("min_observations", 50)
    if type(min_obs) is not int or min_obs < 10:
        raise ValueError("Eclipse features require at least ten observations")
    if not hasattr(periodfind, "FourierDecomposition") or not hasattr(periodfind, "set_device"):
        raise ImportError("Use the CPU-capable periodfind build documented in PHOTOMETRY.md")
    periodfind.set_device("cpu")
    curves = light_curves(read_table(photometry))
    output = Path(output)
    if any(
        Path(p).resolve() in {(output / "features.parquet").resolve(), (output / "peaks.parquet").resolve()}
        for p in (photometry, cached_peaks)
        if p is not None
    ):
        raise ValueError("Choose an output directory separate from input tables")
    cache = None
    baseline, grid_count = None, None
    if cached_peaks is not None:
        cache = read_table(cached_peaks)
        required = {"source_id", "band", "curve_sha256", *peak_columns(config)}
        if not required.issubset(cache) or cache.duplicated(["source_id", "band"]).any():
            raise ValueError(
                "Cached peaks require unique source/band keys, curve hashes and all requested peaks"
            )
        cache = cache.assign(source_id=cache.source_id.astype(str)).set_index(["source_id", "band"])
        for curve in curves:
            key = (curve.source_id, curve.band)
            if key not in cache.index or cache.loc[key, "curve_sha256"] != curve_digest(curve):
                raise ValueError(f"Cached peaks do not correspond to the supplied photometry: {key}")
    else:
        eligible = [np.ptp(c.time) for c in curves if c.n_obs >= min_obs]
        observed = float(max(eligible, default=0))
        baseline = observed if baseline_days is None else float(baseline_days)
        if baseline < observed:
            raise ValueError("The fixed grid baseline cannot be shorter than a searched light curve")
        if observed > 0:
            preflight_grid(config, baseline, max_trials)
            fresh = PeriodExtractor(config)
            fresh._baseline = baseline
            fresh._grid = fresh._build_freq_grid(baseline)
            grid_count = len(fresh._grid)
        else:
            fresh = None
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
            extractor = (
                fresh
                if cache is None
                else CachedExtractor(config, cache.loc[(curve.source_id, curve.band), :])
            )
            values, peaks, state = compute(curve, extractor, min_obs)
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
        for kind, columns in (("features", FEATURES), ("peaks", peak_columns(config))):
            table = pd.DataFrame(
                [{**{k: r[k] for k in identifiers}, **r[kind]} for r in records],
                columns=[*identifiers, *columns],
            )
            for name in columns:
                table[name] = pd.to_numeric(table[name]).astype(float)
            write_parquet(output / f"{kind}.parquet", table)
        write_json(output / "schema.json", dict(name="wdb_period_eclipse50_v1", columns=list(FEATURES)))
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
        write_json(output / "status.json", status)
        return status
