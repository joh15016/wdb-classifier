"""Full-table sklearn tree fits with portable, explicit preprocessing state."""

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def matrix(values):
    values = np.asarray(values, dtype=np.float32).astype(float)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    values[~np.isfinite(values)] = np.nan
    return values


def transform(state, values):
    values = matrix(values)
    if values.shape[1] != len(state["schema"]["columns"]):
        raise ValueError("Feature width differs from the fitted schema")
    missing = np.isnan(values)
    return np.c_[np.where(missing, state["medians"], values), missing[:, state["missing_columns"]]]


def fit_tree(values, labels, schema, *, seed, family="random_forest", parameters=None):
    values, labels = matrix(values), np.asarray(labels)
    if (
        values.shape[1] != len(schema.columns)
        or len(values) != len(labels)
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError("Aligned features, schema and both binary classes are required")
    parameters = dict(parameters or {})
    if {"random_state", "n_jobs", "warm_start"}.intersection(parameters):
        raise ValueError("Seed, one CPU worker and full-table fitting are managed by the workflow")
    defaults = dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt")
    if family == "random_forest":
        defaults["class_weight"] = "balanced_subsample"
        estimator_class = RandomForestClassifier
    elif family == "extra_trees":
        defaults["class_weight"] = "balanced"
        estimator_class = ExtraTreesClassifier
    else:
        raise ValueError(f"Unsupported tree family: {family}")
    defaults.update(parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nan_to_num(np.nanmedian(values, axis=0), nan=0.0)
    state = dict(
        format_version=1,
        family=family,
        schema=schema.as_dict(),
        seed=int(seed),
        parameters=defaults,
        medians=medians,
        missing_columns=np.flatnonzero(np.isnan(values).any(axis=0)),
    )
    estimator = estimator_class(**defaults, random_state=seed, n_jobs=1)
    estimator.fit(transform(state, values), labels)
    state["estimator"] = estimator
    return state


def predict_score(state, values):
    estimator = state["estimator"]
    positive = list(estimator.classes_).index(1)
    score = estimator.predict_proba(transform(state, values))[:, positive]
    # Quantize RF scores consistently for serialization and threshold comparisons.
    if state["family"] == "random_forest":
        score = score.astype(np.float32)
    return np.asarray(score, float)
