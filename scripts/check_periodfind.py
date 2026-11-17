"""Small synthetic CPU backend check, never a dense period search."""

import json
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402
import periodfind  # noqa: E402


def main():
    periodfind.set_device("cpu")
    rng = np.random.default_rng(12)
    times = [np.sort(rng.uniform(0, 5, 96)).astype(np.float32)]
    mags = [np.sin(2 * np.pi * times[0] / 0.2).astype(np.float32)]
    errs = [np.full(96, 0.1, dtype=np.float32)]
    periods = np.linspace(0.18, 0.22, 32, dtype=np.float32)
    for algorithm in (periodfind.LombScargle(), periodfind.ConditionalEntropy(), periodfind.AOV()):
        values = algorithm.calc(times, mags, periods, np.zeros(1, dtype=np.float32), output="periodogram")
        if len(values) != 1 or not np.isfinite(np.asarray(values[0].data)).all():
            raise RuntimeError("Backend returned nonfinite synthetic periodogram values")
    fits = periodfind.FourierDecomposition(device="cpu").calc(
        times, mags, errs, np.array([0.2], dtype=np.float32)
    )
    if np.asarray(fits).shape != (1, 14) or not np.isfinite(fits).all():
        raise RuntimeError("Backend Fourier API failed the synthetic smoke check")
    print(json.dumps(dict(backend="cpu", algorithms=["LS", "CE", "AOV"], fourier_fields=14, complete=True)))


if __name__ == "__main__":
    main()
