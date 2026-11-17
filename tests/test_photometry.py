import json

import numpy as np
import pandas as pd
import pytest

from wdb_classifier.photometry.contracts import PeriodConfig, light_curves
from wdb_classifier.photometry.ztf import ZTFConfig, fetch


def test_time_precision_and_band_policy():
    frame = pd.DataFrame(
        dict(
            source_id=["1"] * 5,
            band=["r", "g", "r", "r", "g"],
            time=2450000.0 + np.array([0.004, 0.003, 0.001, 0.002, 0.007]),
            mag=[18.0] * 5,
            mag_err=[0.1, 0.1, 0.1, 0.0, 0.1],
        )
    )
    (curve,) = light_curves(frame)
    assert curve.band == "r" and curve.n_obs == 2
    assert np.isclose(np.ptp(curve.time), 0.003)
    with pytest.raises(ValueError, match="float32"):
        light_curves(frame.assign(time=frame.time.astype(np.float32)))


class MockKowalski:
    def __init__(self, failed=False):
        self.calls = []
        self.failed = failed

    def query(self, *, queries, use_batch_query, max_n_threads):
        assert use_batch_query and max_n_threads == 1
        ids = queries[0]["query"]["filter"]["_id"]["$in"]
        self.calls.append(ids)
        if self.failed:
            return {"kowalski": [{"status": "error", "message": "sensitive details"}]}
        observations = [
            dict(hjd=2450000.0 + dt, mag=18.0, magerr=err, catflags=flag, programid=program)
            for dt, err, flag, program in [
                (0.0, 0.1, 0, 1),
                (0.001, 0.1, 0, 1),
                (0.01, 0.1, 1, 1),
                (0.02, 0.1, 0, 2),
                (0.03, 0.0, 0, 1),
                (0.04, 0.1, 0, 1),
            ]
        ]
        docs = [
            dict(_id=i, filter=1 if i == 11 else 2, ra=10.0, dec=20.0, data=observations)
            for i in ids
            if i != 13
        ]
        return {"kowalski": [{"status": "success", "data": docs}]}


def test_fetch_resumes_and_distinguishes_filtered_from_empty(tmp_path):
    ids = tmp_path / "ids.csv"
    ids.write_text("ztf_id,label\n11,1\n12,0\n13,0\n")
    config = ZTFConfig("example.org", "ZTF_sources_20240515", bands=("g",), program_ids=(1,))
    client = MockKowalski()
    out = tmp_path / "retrieve"
    assert not fetch(ids, config, out, max_new_sources=1, client=client)["complete"]
    assert fetch(ids, config, out, max_new_sources=2, client=client)["complete"]
    assert client.calls == [[11], [12, 13]]
    assert fetch(ids, config, out, client=client)["new_sources"] == 0
    data = pd.read_parquet(out / "photometry.parquet")
    assert list(data.source_id) == ["11", "11"] and np.isclose(np.ptp(data.time), 0.04)
    assert list(pd.read_parquet(out / "retrieval_audit.parquet").status) == [
        "ok",
        "no_clean_epochs",
        "no_matching_document",
    ]


def test_fetch_failure_is_not_cached(tmp_path, monkeypatch):
    from wdb_classifier.photometry import ztf

    monkeypatch.setattr(ztf.time, "sleep", lambda _: None)
    ids = tmp_path / "ids.csv"
    ids.write_text("source_id\n11\n")
    config = ZTFConfig("example.org", "ZTF_sources_20240515")
    client = MockKowalski(failed=True)
    with pytest.raises(RuntimeError, match="Batch not cached") as error:
        fetch(ids, config, tmp_path / "out", client=client)
    assert "sensitive details" not in str(error.value)
    assert len(client.calls) == 3
    assert not list((tmp_path / "out/sources").glob("*.json"))


def test_fresh_extraction_and_cached_replay(tmp_path):
    pytest.importorskip("periodfind")
    pytest.importorskip("astropy")
    from wdb_classifier.photometry.extract import FEATURES, preflight_grid, run

    with pytest.raises(ValueError, match="limit"):
        preflight_grid(PeriodConfig(), 2000.0, 500)
    rng = np.random.default_rng(19)
    t = np.sort(rng.uniform(0, 8, 240))
    m = 18 + 0.4 * (((t + 0.035) % 0.14) < 0.012) + rng.normal(0, 0.02, len(t))
    phot = tmp_path / "phot.parquet"
    frame = pd.DataFrame(dict(source_id="11", band="g", time=2450000.0 + t, mag=m, mag_err=0.02))
    frame.to_parquet(phot, index=False)
    cfg = tmp_path / "period.toml"
    cfg.write_text(
        '[period]\nalgorithms=["LS","CE","AOV"]\nmin_period_days=.08\nmax_period_days=.25\n'.replace(
            "=.", "=0."
        )
    )
    fresh, replay = tmp_path / "fresh", tmp_path / "replay"
    assert run(phot, cfg, fresh)["successful"] == 1
    assert run(phot, cfg, replay, cached_peaks=fresh / "peaks.parquet")["successful"] == 1
    a, b = (pd.read_parquet(d / "features.parquet") for d in (fresh, replay))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert len(FEATURES) == 50 and np.isfinite(a.eclipse_best_depth_snr.iloc[0])
    assert json.loads((fresh / "schema.json").read_text())["name"] == "wdb_period_eclipse50_v1"
    checkpoint = replay / "sources/00000000.json"
    before = checkpoint.stat().st_mtime_ns
    assert run(phot, cfg, replay, cached_peaks=fresh / "peaks.parquet")["new_sources"] == 0
    assert checkpoint.stat().st_mtime_ns == before
    assert json.loads((replay / "manifest.json").read_text())["grid_count"] is None
    frame.loc[0, "mag"] += 0.01
    frame.to_parquet(phot, index=False)
    with pytest.raises(ValueError, match="do not correspond"):
        run(phot, cfg, tmp_path / "mismatch", cached_peaks=fresh / "peaks.parquet")
