"""ZTF retrieval and CPU period/Fourier/eclipse feature commands."""

import argparse
import json
import os
import resource
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="python -m wdb_classifier.photometry")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="Retrieve numeric ZTF source IDs from Kowalski")
    for name in ("ids", "config", "output"):
        fetch_parser.add_argument(f"--{name}", type=Path, required=True)
    fetch_parser.add_argument("--max-new-sources", type=int, default=20)
    fetch_parser.add_argument("--token-env", default="WDB_ZTF_TOKEN")
    extract_parser = commands.add_parser("extract", help="Compute period, Fourier and eclipse features")
    for name in ("photometry", "config", "output"):
        extract_parser.add_argument(f"--{name}", type=Path, required=True)
    extract_parser.add_argument("--cached-peaks", type=Path)
    extract_parser.add_argument("--baseline-days", type=float)
    extract_parser.add_argument("--max-new-sources", type=int, default=1)
    extract_parser.add_argument("--max-trials", type=int, default=7_000_000)
    extract_parser.add_argument("--seconds", type=float, default=120)
    extract_parser.add_argument("--threads", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["RAYON_NUM_THREADS"] = str(getattr(args, "threads", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4 * 1024**3
    if soft == resource.RLIM_INFINITY or soft > cap:
        resource.setrlimit(
            resource.RLIMIT_AS, (cap if hard == resource.RLIM_INFINITY else min(cap, hard), hard)
        )
    if args.command == "fetch":
        from .ztf import ZTFConfig, fetch

        result = fetch(
            args.ids,
            ZTFConfig.load(args.config),
            args.output,
            max_new_sources=args.max_new_sources,
            token_env=args.token_env,
        )
    else:
        from .extract import run

        result = run(
            args.photometry,
            args.config,
            args.output,
            cached_peaks=args.cached_peaks,
            max_new_sources=args.max_new_sources,
            baseline_days=args.baseline_days,
            max_trials=args.max_trials,
            seconds=args.seconds,
            threads=args.threads,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
