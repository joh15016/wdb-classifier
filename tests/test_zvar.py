"""Generated ZVAR fixtures only: no credentials or survey network requests."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wdb_classifier.photometry.contracts import PeriodConfig
from wdb_classifier.photometry.zvar import (
    FPW_FEATURES,
    ZVARConfig,
    fetch,
    fpw_features,
    load_candidates,
    separation_arcsec,
)


def photometry(tmp_path, count=3):
    rng = np.random.default_rng(22)
    time = np.sort(rng.uniform(0, 8, 120))
    frames = [
        pd.DataFrame(
            dict(
                source_id=str(11 + i),
                band="g",
                ra=10.0 + i,
                dec=20.0,
                time=2450000.0 + time,
                mag=18 + 0.3 * (((time + 0.01) % 0.14) < 0.012),
                mag_err=0.03,
            )
        )
        for i in range(count)
    ]
    frame = pd.concat(frames, ignore_index=True)
    path = tmp_path / "photometry.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def period(**overrides):
    return (
        dict(
            band="g",
            field=309,
            ccd=5,
            quad=3,
            period=0.14,
            significance=12.0,
            bin=20,
            duplicate_freq=False,
            fap=0.001,
            n_sigma=3.1,
        )
        | overrides
    )


class Session:
    def __init__(self, *, unmatched=(), ambiguous=(), periods=None, status=200, malformed=False, error=None):
        self.calls = []
        self.unmatched, self.ambiguous = unmatched, ambiguous
        self.periods = [period()] if periods is None else periods
        self.status, self.malformed, self.error = status, malformed, error

    def post(self, url, *, json, timeout, allow_redirects):
        assert url == ZVARConfig().base_url + "/cone_search"
        assert timeout[0] == 5 and allow_redirects is False
        self.calls.append(json)
        if self.error:
            raise self.error
        data = {}
        for p in json["positions"]:
            sid = p["name"]
            assert set(p) == {"name", "ra", "dec", "radius"}
            obj = dict(psid=str(10000000000000000 + int(sid)), ra=p["ra"], dec=p["dec"], periods=self.periods)
            matches = [] if sid in self.unmatched else [obj]
            if sid in self.ambiguous:
                matches.append(dict(obj, psid=str(int(obj["psid"]) + 1000), ra=p["ra"] + 0.0001))
            data[sid] = matches
        if self.malformed:
            data = {"unexpected": []}
        return SimpleNamespace(status_code=self.status, json=lambda: data)


def test_zvar_fetch_resume_and_match_states(tmp_path, monkeypatch):
    path, frame = photometry(tmp_path)
    frame["label"] = [1] * len(frame)
    frame["known_period_days"] = 123.0
    frame.to_parquet(path, index=False)
    session = Session(unmatched=("12",), ambiguous=("13",))
    out = tmp_path / "zvar"
    assert not fetch(path, ZVARConfig(), out, max_new_sources=1, session=session)["complete"]
    assert fetch(path, ZVARConfig(), out, max_new_sources=2, session=session)["complete"]
    assert len(session.calls) == 2
    monkeypatch.delenv("ZVAR_USERNAME", raising=False)
    monkeypatch.delenv("ZVAR_PASSWORD", raising=False)
    before = (out / "sources/11.json").stat().st_mtime_ns
    assert fetch(path, ZVARConfig(), out)["new_sources"] == 0
    assert (out / "sources/11.json").stat().st_mtime_ns == before
    assert pd.read_parquet(out / "match_audit.parquet").status.tolist() == [
        "matched",
        "no_match",
        "ambiguous_match",
    ]
    selected, fingerprint = load_candidates(out, frame, PeriodConfig())
    assert len(selected["11"]["candidates"]) == 1
    assert not selected["12"]["candidates"] and not selected["13"]["candidates"]
    assert fingerprint["files"]["11"]
    for checkpoint in (out / "sources").glob("*.json"):
        data = checkpoint.read_text()
        assert "label" not in data and "known_period" not in data


@pytest.mark.parametrize(
    "options",
    [
        dict(status=401),
        dict(status=500),
        dict(malformed=True),
        dict(error=RuntimeError("private-response-text")),
        dict(periods=[period(period=-1)]),
    ],
)
def test_failed_zvar_queries_are_not_cached(tmp_path, options):
    path, _ = photometry(tmp_path)
    out = tmp_path / "zvar"
    with pytest.raises((ValueError, RuntimeError)) as error:
        fetch(path, ZVARConfig(), out, session=Session(**options))
    assert "private-response-text" not in str(error.value)
    assert not list((out / "sources").glob("*.json"))


def test_missing_coordinates_and_zero_budget_need_no_credentials(tmp_path, monkeypatch):
    path, frame = photometry(tmp_path)
    monkeypatch.delenv("ZVAR_USERNAME", raising=False)
    monkeypatch.delenv("ZVAR_PASSWORD", raising=False)
    assert fetch(path, ZVARConfig(), tmp_path / "zero", max_new_sources=0)["completed"] == 0
    frame = frame.drop(columns=["ra", "dec"])
    frame.to_parquet(path, index=False)
    out = tmp_path / "no_coordinates"
    assert fetch(path, ZVARConfig(), out)["complete"]
    selected, _ = load_candidates(out, frame, PeriodConfig())
    assert all(r["status"] == "missing_coordinates" and not r["candidates"] for r in selected.values())


def test_candidate_units_ranking_flags_and_band_policy(tmp_path):
    path, frame = photometry(tmp_path)
    candidates = [
        period(significance=100, bin=5, fap=0.5),
        period(band="r", period=0.07, significance=9, bin=10, fap=0.0001),
        period(period=11, fap=0.00001),
        period(duplicate_freq=True, fap=0.000001),
    ]
    out = tmp_path / "zvar"
    fetch(path, ZVARConfig(), out, session=Session(periods=candidates))
    selected, _ = load_candidates(out, frame, PeriodConfig())
    rows = selected["11"]["candidates"]
    assert len(rows) == 2 and rows[0]["band"] == "r" and rows[0]["period"] == 0.07
    features = fpw_features(rows)
    assert features["zvar_fpw_significance"] == 9 and features["zvar_fpw_available"] == 1
    assert set(features) == set(FPW_FEATURES)
    same, _ = load_candidates(out, frame, PeriodConfig(), band_policy="same_band")
    assert len(same["11"]["candidates"]) == 1 and same["11"]["candidates"][0]["band"] == "g"
    frame["ra"] += 0.1
    with pytest.raises(ValueError, match="coordinates do not correspond"):
        load_candidates(out, frame, PeriodConfig())


def test_positions_and_endpoint_safety(tmp_path):
    assert separation_arcsec(359.9999, 0, 0.0001, 0) == pytest.approx(0.72)
    with pytest.raises(ValueError, match="HTTPS ZVAR endpoint"):
        ZVARConfig(base_url="https://example.org")
    with pytest.raises(ValueError, match="radius"):
        ZVARConfig(accept_radius_arcsec=4)
    path, frame = photometry(tmp_path)
    frame.loc[0, "ra"] += 0.01
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Inconsistent coordinates"):
        fetch(path, ZVARConfig(), tmp_path / "zvar", session=Session())


def setup_extraction(tmp_path):
    pytest.importorskip("periodfind")
    pytest.importorskip("astropy")
    path, frame = photometry(tmp_path)
    config = tmp_path / "period.toml"
    config.write_text('[period]\nalgorithms=["LS","CE","AOV"]\nmin_period_days=0.08\nmax_period_days=0.25\n')
    return path, frame, config


def test_zvar_preferred_features_never_run_a_local_grid(tmp_path, monkeypatch):
    path, _, config = setup_extraction(tmp_path)
    from wdb_classifier.photometry.extract import run
    from wdb_classifier.photometry.period import PeriodExtractor

    cache, out = tmp_path / "zvar", tmp_path / "features"
    fetch(path, ZVARConfig(), cache, session=Session())

    def forbidden(*args, **kwargs):
        raise AssertionError("ZVAR-backed source must not allocate/search a period grid")

    monkeypatch.setattr(PeriodExtractor, "_build_freq_grid", forbidden)
    monkeypatch.setattr(PeriodExtractor, "_run_algorithm", forbidden)
    status = run(path, config, out, zvar_cache=cache, zvar_mode="prefer", max_new_sources=3, max_trials=1)
    assert status["external_only_sources"] == 3 and status["grid_count"] is None
    features = pd.read_parquet(out / "features.parquet")
    assert features.period_provider.eq("zvar_fpw").all()
    assert features.period_significance.isna().all()  # no FPW/local significance substitution
    assert features.zvar_fpw_period.eq(0.14).all() and np.isfinite(features.f1_power).all()
    assert pd.read_parquet(out / "peaks.parquet").empty
    assert len(pd.read_parquet(out / "zvar_candidates.parquet")) == 3
    schema = json.loads((out / "schema.json").read_text())
    assert schema["name"] == "wdb_period_eclipse_zvar_prefer57_v1" and len(schema["columns"]) == 57
    checkpoint = out / "sources/00000000.json"
    before = checkpoint.stat().st_mtime_ns
    assert run(path, config, out, zvar_cache=cache, zvar_mode="prefer", max_trials=1)["new_sources"] == 0
    assert checkpoint.stat().st_mtime_ns == before


def test_zvar_fallback_and_supplement_reuse_existing_local_peaks(tmp_path, monkeypatch):
    path, _, config = setup_extraction(tmp_path)
    from wdb_classifier.photometry.extract import run
    from wdb_classifier.photometry.period import PeriodExtractor

    local, cache = tmp_path / "local", tmp_path / "zvar"
    assert run(path, config, local, max_new_sources=3)["successful"] == 3
    fetch(path, ZVARConfig(), cache, session=Session(unmatched=("12",), ambiguous=("13",)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reuse peaks, not search a fresh grid")

    monkeypatch.setattr(PeriodExtractor, "_build_freq_grid", forbidden)
    preferred = tmp_path / "prefer"
    status = run(
        path,
        config,
        preferred,
        zvar_cache=cache,
        zvar_mode="prefer",
        cached_peaks=local / "peaks.parquet",
        max_new_sources=3,
    )
    assert status["external_only_sources"] == 1 and status["successful"] == 3
    f = pd.read_parquet(preferred / "features.parquet").set_index("source_id")
    assert f.loc["11", "period_provider"] == "zvar_fpw"
    assert f.loc["12", "period_provider"] == "cached_local_peaks" and f.loc["12", "zvar_status"] == "no_match"
    assert f.loc["13", "zvar_status"] == "ambiguous_match"
    assert len(pd.read_parquet(preferred / "peaks.parquet")) == 2
    supplemental = tmp_path / "supplement"
    run(path, config, supplemental, zvar_cache=cache, cached_peaks=local / "peaks.parquet", max_new_sources=3)
    f = pd.read_parquet(supplemental / "features.parquet")
    assert f.period_provider.eq("cached_local_peaks").all()
    pd.testing.assert_series_equal(f.period, pd.read_parquet(local / "features.parquet").period)
    assert f.zvar_fpw_available.sum() == 1


def test_changed_zvar_cache_cannot_silently_resume(tmp_path):
    path, _, config = setup_extraction(tmp_path)
    from wdb_classifier.photometry.extract import run

    cache, out = tmp_path / "zvar", tmp_path / "features"
    fetch(path, ZVARConfig(), cache, session=Session())
    run(path, config, out, zvar_cache=cache, zvar_mode="prefer", max_new_sources=3)
    checkpoint = cache / "sources/11.json"
    value = json.loads(checkpoint.read_text())
    value["matches"][0]["periods"][0]["period"] = 0.15
    checkpoint.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        run(path, config, out, zvar_cache=cache, zvar_mode="prefer")
