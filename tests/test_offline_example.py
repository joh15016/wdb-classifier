import runpy
from pathlib import Path


def test_offline_example_and_resume(tmp_path, monkeypatch):
    for name in ("WDB_ZTF_TOKEN", "WDB_ZTF_USERNAME", "WDB_ZTF_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    example = Path(__file__).resolve().parents[1] / "examples/offline_demo.py"
    main = runpy.run_path(str(example))["main"]
    first = main(["--output", str(tmp_path / "demo")])
    assert first["engine"] == "Hyrax" and first["new_fits"] == 9 and first["unseen_sources"] == 12
    assert not first["real_data_or_credentials_required"]
    checkpoint = tmp_path / "demo/model/folds/0/bundle.joblib"
    before = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
    second = main(["--output", str(tmp_path / "demo")])
    assert second["new_fits"] == 0 and second["reused_predictions"]
    assert before == (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns)
