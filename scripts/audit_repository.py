"""Audit prospective Git files without printing secret values or probing APIs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PRIVATE_SUFFIXES = {
    ".pem",
    ".key",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".parquet",
    ".h5",
    ".hdf5",
}
PUBLIC_REVISION = "e96c6e1ffa9ca8f3ade633e800e60a6256006102"


def audit(root):
    root = Path(root).resolve()
    paths = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    paths = sorted({name for name in paths if name and (root / name).is_file()})
    problems = []
    for name in paths:
        path = root / name
        if root not in path.resolve().parents:
            problems.append(dict(file=name, category="symlink outside repository"))
            continue
        if (
            path.parts[len(root.parts)] in {".venv", "data", "runs"}
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name == "environment.json"
            or path.name.lower().startswith(("credentials", "secrets"))
        ):
            problems.append(dict(file=name, category="private/data/model artifact"))
        if path.stat().st_size > 1_000_000:
            problems.append(dict(file=name, category="file larger than source-review limit"))
    if problems:
        return dict(complete=False, files=len(paths), artifact_findings=problems, secret_findings=[])
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "detect_secrets",
            "scan",
            "--all-files",
            "--no-verify",
            "--exclude-secrets",
            f"^{PUBLIC_REVISION}$",
            *paths,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError("Secret scanner failed; install the dev extra. No raw scanner output displayed.")
    results = json.loads(process.stdout)["results"]
    findings = [
        dict(file=filename, line=item["line_number"], category=item["type"])
        for filename, items in results.items()
        for item in items
    ]
    return dict(
        complete=not findings,
        files=len(paths),
        artifact_findings=[],
        secret_findings=findings,
        secret_verification_network_calls=False,
        note="Heuristic scan, not a guarantee; review the staged diff before committing.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
