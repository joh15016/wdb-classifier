"""Normalize a local cached photometry table without querying any survey."""

import argparse
from pathlib import Path

from wdb_classifier.artifacts import digest, write_json
from wdb_classifier.data import read_table


def run(source, output, time_column, time_system):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix(".import.json").exists():
        raise FileExistsError("Choose a new output; existing photometry is preserved")
    if output.suffix != ".parquet":
        raise ValueError("Use a Parquet output")
    frame = read_table(source).rename(columns={time_column: "time"})
    required = ["source_id", "band", "time", "mag", "mag_err"]
    if not set(required).issubset(frame):
        raise ValueError(f"Required photometry columns: {required}")
    # Preserve positions and other optional metadata, while normalizing the required fields.
    frame["source_id"] = frame.source_id.astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(
        output.with_suffix(".import.json"),
        dict(
            source=str(source.resolve()),
            source_sha256=digest(source),
            output_sha256=digest(output),
            rows=len(frame),
            sources=frame.source_id.nunique(),
            time_system=time_system,
            operation="column normalization only; no time or flux conversions",
        ),
    )
    print(f"Imported {len(frame)} measurements for {frame.source_id.nunique()} sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--time-system", required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.time_column, args.time_system)
