"""Atomic artifacts, input fingerprints and a single-writer run lock."""

import fcntl
import hashlib
import importlib.metadata
import json
from contextlib import contextmanager
from pathlib import Path

import joblib


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_model(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(state, temporary)
    temporary.replace(path)


def guard(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Incompatible checkpoint: {path}. Choose a new output directory.")
    else:
        write_json(path, value)


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "hyrax")
    }


def code_fingerprint():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
