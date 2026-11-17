"""Bounded full-table tree fitting; no neural minibatches or dummy optimizers."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .artifacts import code_fingerprint, digest, guard, run_lock, versions, write_json, write_model
from .config import TrainingConfig
from .data import FeatureSchema, load_features, read_table
from .evaluation import metrics, outer_splits, splits, threshold
from .models import fit_tree, predict_score


class BudgetStop(Exception):
    """Safe stop between completed fits, not a mid-fit optimizer checkpoint."""


def _fit(path, x, frame, train, config, schema, seed, budget):
    record_path = path / "complete.json"
    spec = dict(training_ids=frame.source_id.iloc[train].tolist(), seed=int(seed))
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["spec"] != spec or digest(path / "model.joblib") != record["sha256"]:
            raise ValueError(f"Changed fitted artifact: {path}")
        return joblib.load(path / "model.joblib")
    if time.monotonic() >= budget["until"] or (
        budget["max_fits"] is not None and budget["new_fits"] >= budget["max_fits"]
    ):
        raise BudgetStop
    budget["new_fits"] += 1
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        state = fit_tree(
            x[train],
            frame.label.iloc[train].to_numpy(),
            schema,
            seed=seed,
            family=config.family,
            parameters=config.parameters,
        )
    state["training_ids"] = spec["training_ids"]
    state["training_groups"] = frame.group_id.iloc[train].tolist()
    write_model(path / "model.joblib", state)
    write_json(
        record_path,
        dict(spec=spec, sha256=digest(path / "model.joblib"), seconds=time.perf_counter() - started),
    )
    return state


def train(config: TrainingConfig, *, minutes=5.0, max_new_fits=None):
    if minutes <= 0 or (max_new_fits is not None and max_new_fits < 0):
        raise ValueError("Positive time budget and nonnegative fit limit required")
    output = Path(config.output)
    started = time.monotonic()
    schema = FeatureSchema.load(config.schema)
    frame, x = load_features(config.features, schema, labeled=True)
    assignments = read_table(config.assignments) if config.assignments else None
    outer = outer_splits(frame, config.outer_folds, config.seed, assignments)
    paths = [config.features, config.schema] + ([config.assignments] if config.assignments else [])
    manifest = dict(
        format_version=1,
        configuration=config.as_dict(),
        schema=schema.as_dict(),
        files={str(p): digest(p) for p in paths},
        code=code_fingerprint(),
        versions=versions(),
        rows=len(frame),
        positives=int(frame.label.sum()),
        source_grouped=True,
        independent_final_test=False,
        scores_are_calibrated_probabilities=False,
        threshold_policy="inner OOF background quantile, higher, nextafter toward infinity",
    )
    budget = dict(until=started + 60 * minutes, max_fits=max_new_fits, new_fits=0)
    with run_lock(output):
        guard(output / "manifest.json", manifest)
        for fold, (train_index, test_index) in enumerate(outer):
            folder = output / f"folds/{fold}"
            if (folder / "complete.json").exists():
                done = json.loads((folder / "complete.json").read_text())
                for name, sha in done["files"].items():
                    if digest(folder / name) != sha:
                        raise ValueError(f"Completed fold artifact changed: {folder / name}")
                print(f"Fold {fold + 1}/{len(outer)}: reused", flush=True)
                continue
            try:
                inner_scores = np.full(len(train_index), np.nan)
                inner_frame = frame.iloc[train_index].reset_index(drop=True)
                for k, (a, b) in enumerate(splits(inner_frame, config.inner_folds, config.seed + fold)):
                    state = _fit(
                        folder / f"inner{k}",
                        x,
                        frame,
                        train_index[a],
                        config,
                        schema,
                        config.seed + 100 * fold + k,
                        budget,
                    )
                    inner_scores[b] = predict_score(state, x[train_index[b]])
                if not np.isfinite(inner_scores).all():
                    raise ValueError("Incomplete inner OOF scores")
                cutoffs = {
                    str(b): threshold(frame.label.iloc[train_index], inner_scores, b) for b in config.budgets
                }
                state = _fit(
                    folder / "outer", x, frame, train_index, config, schema, config.seed + fold, budget
                )
                state["thresholds"] = cutoffs
                # Separate inference bundle; the resumable fitted model stays immutable.
                write_model(folder / "bundle.joblib", state)
                score = predict_score(state, x[test_index])
                parts = [
                    pd.DataFrame(
                        dict(
                            source_id=frame.source_id.iloc[test_index].to_numpy(),
                            label=frame.label.iloc[test_index].to_numpy(),
                            fold=fold,
                            score=score,
                            budget=b,
                            threshold=cutoffs[str(b)],
                            predicted_label=(score >= cutoffs[str(b)]).astype(int),
                        )
                    )
                    for b in config.budgets
                ]
                pd.concat(parts, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
                write_json(folder / "thresholds.json", cutoffs)
                write_json(
                    folder / "split_ids.json",
                    dict(
                        training=frame.source_id.iloc[train_index].tolist(),
                        test=frame.source_id.iloc[test_index].tolist(),
                    ),
                )
                write_json(
                    folder / "complete.json",
                    dict(
                        files={
                            p: digest(folder / p)
                            for p in ("bundle.joblib", "predictions.csv", "thresholds.json", "split_ids.json")
                        }
                    ),
                )
                print(f"Fold {fold + 1}/{len(outer)}: completed; new fits={budget['new_fits']}", flush=True)
            except BudgetStop:
                break
        completed = [k for k in range(len(outer)) if (output / f"folds/{k}/complete.json").exists()]
        complete = len(completed) == len(outer)
        if complete:
            predicted = pd.concat(
                [
                    pd.read_csv(
                        output / f"folds/{k}/predictions.csv",
                        dtype={"source_id": str},
                        float_precision="round_trip",
                    )
                    for k in completed
                ],
                ignore_index=True,
            )
            predicted.to_csv(output / "oof_predictions.csv", index=False)
            summary = [
                dict(budget=float(b), **metrics(p.label, p.score, p.predicted_label))
                for b, p in predicted.groupby("budget")
            ]
            write_json(output / "metrics.json", summary)
        status = dict(
            complete=complete,
            completed_folds=completed,
            new_fits=budget["new_fits"],
            seconds=time.monotonic() - started,
        )
        write_json(output / "status.json", status)
    return status


def predict(run, features, output, *, budget=0.05, allow_training_overlap=False):
    """Fold-vote candidate selection; scores are not calibrated population probabilities."""
    run, output = Path(run), Path(output)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["versions"] != versions():
        raise ValueError("Inference dependency versions differ from the fitted artifacts")
    if not json.loads((run / "status.json").read_text())["complete"]:
        raise ValueError("Complete all training folds before inference")
    schema = FeatureSchema(manifest["schema"]["name"], tuple(manifest["schema"]["columns"]))
    frame, x = load_features(features, schema)
    scores, flags, known_ids, known_groups = [], [], set(), set()
    for k in range(manifest["configuration"]["outer_folds"]):
        folder = run / f"folds/{k}"
        record = json.loads((folder / "complete.json").read_text())
        if digest(folder / "bundle.joblib") != record["files"]["bundle.joblib"]:
            raise ValueError("Inference model checksum changed")
        state = joblib.load(folder / "bundle.joblib")
        if str(budget) not in state["thresholds"]:
            raise ValueError("Choose a budget evaluated during training")
        known_ids.update(state["training_ids"])
        known_groups.update(state["training_groups"])
        score = predict_score(state, x)
        scores.append(score)
        flags.append(score >= state["thresholds"][str(budget)])
    overlap = frame.source_id.isin(known_ids) | frame.group_id.isin(known_groups)
    if overlap.any() and not allow_training_overlap:
        raise ValueError(
            "Inference sources overlap training; use OOF predictions for evaluation, or explicitly allow inspection"
        )
    votes = np.asarray(flags).sum(axis=0)
    selected = votes >= len(flags) // 2 + 1
    result = pd.DataFrame(
        dict(
            source_id=frame.source_id,
            score=np.mean(scores, axis=0),
            fold_votes=votes,
            predicted_label=selected.astype(int),
            label_name=np.where(selected, "WDB candidate", "Not flagged"),
            training_overlap=overlap,
            budget=budget,
            schema=schema.name,
        )
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        result.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        result.to_csv(output, index=False)
    else:
        raise ValueError("Prediction output must be .parquet or .csv")
    return dict(output=str(output), sources=len(result), candidates=int(selected.sum()))
