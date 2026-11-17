"""Portable feature contracts, deliberately independent of any ML framework."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESERVED = {"source_id", "ztf_id", "group_id", "label", "fold", "known_period_days", "name"}


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    columns: tuple[str, ...]

    def __post_init__(self):
        if (
            not self.name
            or not self.columns
            or len(set(self.columns)) != len(self.columns)
            or RESERVED.intersection(self.columns)
        ):
            raise ValueError("Feature names must be unique and exclude IDs, labels and catalog truth")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        return cls(str(value["name"]), tuple(value["columns"]))

    def as_dict(self):
        return dict(name=self.name, columns=list(self.columns))


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"source_id": str, "group_id": str})
    raise ValueError("Use a .parquet or .csv table")


def validate_features(frame, schema, *, labeled=False):
    frame = frame.copy()
    if frame.empty or "source_id" not in frame or frame.source_id.isna().any():
        raise ValueError("A nonempty table with source_id is required")
    if pd.api.types.is_float_dtype(frame.source_id):
        raise ValueError("Source IDs must be strings or integers, never floating point")
    frame["source_id"] = frame.source_id.astype(str)
    if not frame.source_id.is_unique or frame.source_id.str.strip().eq("").any():
        raise ValueError("source_id must be nonempty and unique; join bands before classification")
    if not set(schema.columns).issubset(frame):
        raise ValueError(f"Missing features: {sorted(set(schema.columns) - set(frame))}")
    # Named columns enforce order; quantize consistently at the float32 input boundary.
    try:
        values = frame[list(schema.columns)].to_numpy(dtype=np.float32).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All model features must be numeric") from exc
    values[~np.isfinite(values)] = np.nan
    if "group_id" not in frame:
        frame["group_id"] = frame.source_id
    if frame.group_id.isna().any() or frame.group_id.astype(str).str.strip().eq("").any():
        raise ValueError("group_id cannot be missing; associate physical sources before splitting")
    frame["group_id"] = frame.group_id.astype(str)
    if labeled:
        if "label" not in frame or not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2:
            raise ValueError("Training requires both binary labels: 0 and 1")
        if frame.groupby("group_id").label.nunique().max() != 1:
            raise ValueError("Conflicting labels within a physical source group")
        frame["label"] = frame.label.astype(int)
    return frame, values


def load_features(path, schema, *, labeled=False):
    return validate_features(read_table(path), schema, labeled=labeled)
