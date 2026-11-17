"""Small user interface; training and inference dispatch through Hyrax verbs."""

import argparse
import json
import os
from pathlib import Path


def main():
    # Keep this command bounded on shared machines; do not change importing users' pools.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4 * 1024**3
    if soft == resource.RLIM_INFINITY or soft > cap:
        resource.setrlimit(
            resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard)
        )
    parser = argparse.ArgumentParser(prog="wdb-classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--minutes", type=float, default=5.0)
    train_parser.add_argument("--max-new-fits", type=int)
    predict_parser = sub.add_parser("predict")
    for name in ("run", "features", "output"):
        predict_parser.add_argument(f"--{name}", type=Path, required=True)
    predict_parser.add_argument("--budget", type=float, default=0.05)
    predict_parser.add_argument("--allow-training-overlap", action="store_true")
    feature_parser = sub.add_parser("features")
    feature_parser.add_argument("--photometry", type=Path, required=True)
    feature_parser.add_argument("--output", type=Path, required=True)
    feature_parser.add_argument(
        "--time-system", required=True, help="E.g. HJD_UTC; values are recorded, not converted"
    )
    args = parser.parse_args()
    if args.command == "train":
        from .config import TrainingConfig
        from .hyrax_plugin import create_workflow

        result = create_workflow(TrainingConfig.load(args.config)).wdb_train(
            minutes=args.minutes, max_new_fits=args.max_new_fits
        )
    elif args.command == "predict":
        from .hyrax_plugin import create_workflow

        result = create_workflow().wdb_predict(
            run=args.run,
            features=args.features,
            output=args.output,
            budget=args.budget,
            allow_training_overlap=args.allow_training_overlap,
        )
    else:
        from .artifacts import write_json
        from .data import read_table
        from .features import extract_statistics

        if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
            raise FileExistsError("Choose a new feature output; existing artifacts are preserved")
        if args.output.suffix != ".parquet":
            raise ValueError("Use .parquet for extracted features")
        table, manifest = extract_statistics(read_table(args.photometry))
        manifest["time_system"] = args.time_system
        from .artifacts import digest

        manifest["input_sha256"] = digest(args.photometry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.output, index=False)
        write_json(args.output.with_suffix(".manifest.json"), manifest)
        result = dict(output=str(args.output), rows=len(table), schema=manifest["schema"])
    print(json.dumps(result, indent=2))
