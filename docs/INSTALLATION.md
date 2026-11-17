# Installation

The tested configuration targets **Python 3.11 on Linux x86_64, CPU only**.
The package uses POSIX file locking and process limits. Other Python versions
are permitted by metadata but have not been established as equivalent; Windows
support is not established.

## Python application

Create an isolated environment in a checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -c constraints-cpu.txt -e '.[features,ztf,eclipse,dev]'
python -m pip check
python examples/offline_demo.py --output runs/offline_demo
```

Installation needs package-index access. The offline example itself requires
no network, credentials, real data or native periodfinding. The constraints
pin key CPU/scientific versions, not every transitive dependency. Do not copy
a virtual environment between machines or enable shared site-packages as a
substitute for installing the dependencies.

### Conda alternative

For a fresh interpreter independent of other applications, use the supplied
project manifest instead of the `venv` commands above:

```bash
conda env create --prefix .venv --file environment.yml
conda activate "$PWD/.venv"
python -m pip check
```

`environment.yml` is intentionally minimal and installs this package with its
documented extras and CPU constraints. It does not export every distribution
from the machine. The optional native backend below is installed separately.

## Optional CPU periodfinding backend

Use the pinned `scope-ml/periodfind` source at revision
`e96c6e1ffa9ca8f3ade633e800e60a6256006102`. It provides the `periodfind` Python
dispatch layer and the `periodfind_cpu` Rust extension. A generic package with
the same version number is not assumed to expose the same API.

Install Git, Rust/Cargo and native build tools, activate the environment, then:

```bash
env CUDAHOME=/dev/null CUDA_HOME=/dev/null CARGO_BUILD_JOBS=1 \
  python -m pip install -c constraints-cpu.txt -r requirements/periodfind-cpu.txt
python scripts/check_periodfind.py
python -m pytest -q tests/test_photometry.py
```

The CUDA variables apply only to this command. At the pinned revision the build
automatically compiles CUDA when it finds a toolkit. A non-directory sentinel
selects its no-CUDA path without modifying the dependency. Rust compilation is
bounded to one job. This procedure requires GitHub and Cargo registry access.

The backend check exercises small synthetic LS/CE/AoV periodograms and a
Fourier fit, not a dense survey search. Use it before expensive extraction.

## Tests and credentials

```bash
python -m pytest -q
python -m ruff check src scripts tests examples
python scripts/audit_repository.py
```

Tests use generated fixtures and no live credentials. Optional statistics/native
tests may skip when their dependencies are unavailable. Inspect skip reasons;
a skipped native test is not a successful backend validation.

See [photometry](../PHOTOMETRY.md) for live retrieval. `.env.example` contains
empty variable names only and is not loaded automatically. Secrets belong in
exported environment variables or explicitly selected private external files.

## HPC/GPU resources

Current project commands run CPU tree models and CPU feature extraction.
Scheduler integration and a tested GPU environment are not supplied. Hyrax's
native neural/GPU capabilities do not automatically accelerate the custom tree
operations. Recreate the environment on the cluster and benchmark a bounded
job before configuring arrays, additional workers or a different GPU build.
