"""Photometry and search contracts for periodic and eclipse feature extraction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PERIOD = 5 / 1440
MAX_PERIOD = 10.0


@dataclass(frozen=True)
class PeriodConfig:
    algorithms: tuple[str, ...] = ("LS", "CE", "AOV")
    min_period_days: float = MIN_PERIOD
    max_period_days: float = MAX_PERIOD
    samples_per_peak: float = 10
    top_n_periods: int = 10
    n_chunks_multiplier: int = 3

    def __post_init__(self):
        object.__setattr__(self, "algorithms", tuple(self.algorithms))
        if (
            not self.algorithms
            or len(set(self.algorithms)) != len(self.algorithms)
            or set(self.algorithms) - {"LS", "CE", "AOV", "MHF", "FPW", "BLS"}
        ):
            raise ValueError("Choose distinct, supported period algorithms")
        if not MIN_PERIOD <= self.min_period_days < self.max_period_days <= MAX_PERIOD:
            raise ValueError(
                "This eclipse feature profile supports a search range within 5 minutes to 10 days"
            )
        if not np.isfinite(self.samples_per_peak) or self.samples_per_peak <= 0:
            raise ValueError("samples_per_peak must be positive and finite")
        if not 1 <= self.top_n_periods <= 30 or not 1 <= self.n_chunks_multiplier <= 10:
            raise ValueError("Use 1–30 retained peaks and a chunk multiplier from 1 to 10")


@dataclass
class LightCurve:
    source_id: str
    band: str
    time: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray

    @property
    def n_obs(self):
        return len(self.time)


def light_curves(table):
    """Clean finite/positive-error epochs, then select the most observed band per ID."""
    required = {"source_id", "band", "time", "mag", "mag_err"}
    if not required.issubset(table) or table.empty:
        raise ValueError(f"Nonempty photometry with columns {sorted(required)} is required")
    if table[["source_id", "band"]].isna().any().any() or pd.api.types.is_float_dtype(table.source_id):
        raise ValueError("Missing bands/IDs or floating-point IDs are not supported")
    if table.time.dtype == np.float32:
        raise ValueError("Absolute epochs must not be rounded to float32 before subtracting their origin")
    if "time_system" in table and (table.time_system.isna().any() or table.time_system.nunique() != 1):
        raise ValueError("A run must have one declared time system")
    selected = []
    for source_id, source in table.groupby("source_id", sort=False):
        bands = []
        for band, part in source.groupby("band", sort=False):
            values = part[["time", "mag", "mag_err"]].to_numpy(dtype=np.float64)
            good = np.isfinite(values).all(axis=1) & (values[:, 2] > 0)
            t, m, e = values[good].T
            order = np.argsort(t, kind="stable")
            bands.append(LightCurve(str(source_id), str(band), t[order], m[order], e[order]))
        selected.append(max(bands, key=lambda lc: lc.n_obs))
    return selected
