"""Optional, versioned magnitude statistics from the light-curve library.

This imports the independent light-curve library, not FLARE. Inputs are observed
magnitudes in one declared time system, per band, over the full supplied baseline.
No clipping, transient horizon, Bazin fit or non-simultaneous color construction.
"""

import importlib.metadata

import numpy as np
import pandas as pd


def make_extractor():
    try:
        import light_curve as lc
    except ImportError as exc:
        raise ImportError("Install wdb-classifier[features] to use the light-curve provider") from exc
    return lc.Extractor(
        lc.Amplitude(),
        lc.AndersonDarlingNormal(),
        lc.BeyondNStd(1.0),
        lc.BeyondNStd(2.0),
        lc.Cusum(),
        lc.EtaE(),
        lc.InterPercentileRange(0.10),
        lc.InterPercentileRange(0.25),
        lc.Kurtosis(),
        lc.Mean(),
        lc.MedianAbsoluteDeviation(),
        lc.ReducedChi2(),
        lc.Skew(),
        lc.StandardDeviation(),
        lc.StetsonK(),
        lc.WeightedMean(),
    )


def extract_statistics(photometry):
    required = {"source_id", "band", "time", "mag", "mag_err"}
    if not required.issubset(photometry):
        raise ValueError(f"Photometry requires {sorted(required)}")
    if photometry.empty or photometry[["source_id", "band"]].isna().any().any():
        raise ValueError("Photometry source IDs and bands cannot be missing")
    if pd.api.types.is_float_dtype(photometry.source_id):
        raise ValueError("Source IDs must be strings or integers, not floats")
    extractor = make_extractor()
    columns = [f"lc_mag_v1_{name}" for name in extractor.names]
    rows = []
    for (sid, band), source in photometry.groupby(["source_id", "band"], sort=False):
        values = source[["time", "mag", "mag_err"]].to_numpy(float)
        valid = np.isfinite(values).all(axis=1) & (values[:, 2] > 0)
        table = pd.DataFrame(values[valid], columns=["time", "mag", "mag_err"]).sort_values(
            "time", kind="stable"
        )
        # Deterministic duplicate handling; do not silently invent extra observations.
        table = table.drop_duplicates("time", keep="first")
        n = len(table)
        row = dict(
            source_id=str(sid),
            band=str(band),
            lc_mag_v1_n_input=len(source),
            lc_mag_v1_n_invalid=int((~valid).sum()),
            lc_mag_v1_n_valid=n,
            lc_mag_v1_n_duplicate=int(valid.sum() - n),
            lc_mag_v1_span_days=float(table.time.max() - table.time.min()) if n else np.nan,
        )
        if n >= 4:
            t = table.time.to_numpy(float)
            t = np.ascontiguousarray(t - t[0])
            result = extractor(
                t,
                np.ascontiguousarray(table.mag.to_numpy(float)),
                np.ascontiguousarray(table.mag_err.to_numpy(float)),
                fill_value=np.nan,
                check=True,
            )
        else:
            result = np.full(len(columns), np.nan)
        row.update({name: float(v) if np.isfinite(v) else np.nan for name, v in zip(columns, result)})
        rows.append(row)
    manifest = dict(
        provider="light-curve",
        version=importlib.metadata.version("light-curve"),
        schema="light_curve_magnitude_statistics_v1",
        input_space="magnitude",
        time_unit="days",
        time_system="caller-declared; do not mix time systems",
        bands="separate rows, never averaged",
        baseline="all supplied epochs",
        sigma_clipping=False,
        duplicate_times="stable first valid observation",
        min_valid_points=4,
        columns=columns,
        catalog_or_label_inputs=False,
    )
    return pd.DataFrame(rows), manifest
