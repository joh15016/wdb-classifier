"""Offline plumbing example with generated toy features, not real WDB recovery."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/offline_demo"))
    args = parser.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    import numpy as np
    import pandas as pd

    from wdb_classifier.artifacts import guard
    from wdb_classifier.config import TrainingConfig
    from wdb_classifier.hyrax_plugin import create_workflow

    output = args.output.resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    names = ["toy_depth_snr", "toy_duty_cycle", "toy_scatter"]

    def frame(count, prefix, labeled):
        label = np.tile([0, 0, 0, 1], count // 4)
        values = rng.normal(size=(count, 3))
        values[:, 0] += label * 3
        values[::9, 2] = np.nan
        result = pd.DataFrame(values, columns=names)
        result.insert(0, "source_id", [f"{prefix}-{i:03d}" for i in range(count)])
        if labeled:
            result["label"] = label
        return result

    for name, table in (("train", frame(96, "toy-train", True)), ("unseen", frame(12, "toy-new", False))):
        path = inputs / f"{name}.parquet"
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), table, check_exact=True)
        else:
            table.to_parquet(path, index=False)
    guard(inputs / "schema.json", dict(name="offline_toy_v1", columns=names))
    config = TrainingConfig(
        features=str(inputs / "train.parquet"),
        schema=str(inputs / "schema.json"),
        output=str(output / "model"),
        seed=42,
        outer_folds=3,
        inner_folds=2,
        budgets=[0.1],
        parameters=dict(n_estimators=16, min_samples_leaf=2),
    )
    workflow = create_workflow(config)
    training = workflow.wdb_train(minutes=1, max_new_fits=9)
    if not training["complete"]:
        raise RuntimeError("Toy fits did not finish within the budget; rerun to resume")
    predictions = output / "predictions.parquet"
    reused_predictions = predictions.exists()
    if not reused_predictions:
        workflow.wdb_predict(
            run=config.output, features=inputs / "unseen.parquet", output=predictions, budget=0.1
        )
    result = pd.read_parquet(predictions)
    summary = dict(
        example="generated toy tables; not a WDB accuracy/completeness measurement",
        engine=training["engine"],
        training_sources=96,
        unseen_sources=len(result),
        new_fits=training["new_fits"],
        reused_predictions=reused_predictions,
        output=str(predictions),
        real_data_or_credentials_required=False,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
