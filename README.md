# WDB Classifier

A **Hyrax-based machine-learning application for finding white dwarf binary
candidates in ZTF photometry**, inspired by the separation of data preparation,
feature extraction and classification in [AppleCiDEr](https://github.com/applecider-ml/applecider)
and [FLARE](https://github.com/applecider-ml/flare).

WDB means a binary containing at least one white dwarf; DWD specifically means
two white dwarfs. The initial scientific focus is eclipsing WDBs. Recovery of
all WDB subtypes, including interacting systems, has not been established.

This is a research application under development, not a released survey catalog
or a pretrained, scientifically validated discovery model. Hyrax provides
workflow integration; this package supplies the WDB-specific scientific logic.
It does not use AppleCiDEr/FLARE pretrained classifiers or require their code.

## What is implemented

- Authenticated ZTF retrieval, quality filtering and reusable per-source caches.
- Period searches, multiple candidate peaks, harmonic/sidereal-alias consensus,
  Fourier measurements and eclipse-specific features.
- Optional reuse of precomputed ZVAR FPW results during feature generation,
  with separate provenance, resumable lookup caches and local-search fallback.
- Optional magnitude statistics from the independent `light-curve` library.
- Random forest and Extra Trees training/inference through registered Hyrax
  operations, without modifying Hyrax or routing trees through a neural optimizer.
- Source-grouped evaluation, training-only imputation and missingness indicators,
  training-selected thresholds and held-out predictions.
- Checkpointed fits, bounded CPU commands and portable Parquet/CSV outputs.

## Installation

The tested CPU setup targets Python 3.11 on Linux x86_64:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -c constraints-cpu.txt -e '.[features,ztf,zvar,eclipse,dev]'
```

See [installation](docs/INSTALLATION.md) for the pinned native periodfinding
backend. The CPU environment does not require a GPU. Dependencies retain their
own licenses; see [notices](NOTICE.md).

Alternatively, `environment.yml` defines a project-specific Conda environment:

```bash
conda env create --prefix .venv --file environment.yml
conda activate "$PWD/.venv"
```

Choose either installation method for a new environment. Do not reuse another
application's environment or enable shared site-packages. The environment file
contains only Python, pip, and this package's declared dependencies;
it is not a dump of unrelated installed software.

## Try the offline example

```bash
python examples/offline_demo.py --output runs/offline_demo
```

The example generates toy **feature tables**, fits through Hyrax and predicts
for disjoint toy source IDs. It requires no credentials, real data or native
periodfinding backend. Run it again to reuse completed fits and predictions.
It demonstrates software operation, not astronomical recovery performance.

## Work with real data

The scientific workflow is retrieval/quality control, feature extraction,
source-level table assembly, training/evaluation and candidate scoring.
Photometry must have `source_id, band, time, mag, mag_err`; epochs are in days
in one declared time system. See [photometry commands](PHOTOMETRY.md).

Training takes a table containing `source_id`, binary `label`, and named numeric
features. Supply `group_id` when multiple records belong to one physical source.
The feature schema is a JSON document such as:

```json
{"name": "my_features_v1", "columns": ["period", "eclipse_selected_depth_snr"]}
```

Adapt `configs/random_forest.toml`; its paths are relative to the config file:

```bash
wdb-classifier train --config configs/random_forest.toml --minutes 5
wdb-classifier predict --run runs/random_forest \
  --features data/new_sources.parquet --output runs/candidates.parquet --budget 0.05
```

The Python interface uses actual Hyrax operations:

```python
from wdb_classifier.config import TrainingConfig
from wdb_classifier.hyrax_plugin import create_workflow

h = create_workflow(TrainingConfig.load("configs/random_forest.toml"))
h.wdb_train(minutes=5)
```

The tree operations are `wdb_train` and `wdb_predict`; native Hyrax `train`
remains the neural-training interface. Normal tree fits use one CPU worker.
Time budgets are checked between fits, not inside a running native calculation.

## Interpreting outputs

Predictions include a score, fold votes, yes/no candidate label and training
overlap flag. A positive is a **WDB candidate**, not a confirmed binary. Scores
are not calibrated population probabilities. A training-selected background
budget is not a survey contamination guarantee. Training-source overlap is
rejected by default; held-out predictions are the appropriate evaluation output.

Caches record inputs, code and package versions. Rerunning a compatible command
reuses completed work. Changed definitions require a separate run directory;
prediction files are never silently overwritten. Only load trusted joblib files.

## Current boundaries

Period/eclipse extraction produces 50 fields; optional magnitude statistics
produce a separate 16-field table plus quality diagnostics. Choose and join
features explicitly using the same source/band policy for training and inference.
Their presence does not define or validate a complete scientific classifier.
The opt-in ZVAR profile adds seven namespaced FPW fields; see
[ZVAR feature generation](PHOTOMETRY.md#reuse-precomputed-zvar-fpw-results).

Gaia associations, astrophysical preselection, automated source-level feature
assembly, calibrated ensembles, independent completeness/contamination studies
and survey-scale HPC orchestration remain future work. No such selection cuts
are silently applied by the current fetcher. See [architecture](ARCHITECTURE.md).

## Development and licensing

```bash
python -m pytest -q
python -m ruff check src scripts tests examples
python -m pip check
python scripts/audit_repository.py
```

Tests use generated fixtures and no live service credentials. Optional-backend
tests skip if the relevant dependency is absent. Data, runs, environments and
credentials stay outside version control. See [validation](docs/VALIDATION.md).

A license for this project's own source has not yet been selected. Contributor
approval and dependency distribution requirements must be resolved before a
public release; see [NOTICE.md](NOTICE.md).
