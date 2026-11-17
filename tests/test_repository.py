import importlib.util
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_have_no_personal_paths():
    for path in (ROOT / "configs").glob("*.toml"):
        assert all(value not in path.read_text() for value in ("/home/", "/Users/"))


def test_test_environment_does_not_share_other_site_packages():
    prefix = Path(sys.prefix).resolve()
    for location in site.getsitepackages():
        assert Path(location).resolve().is_relative_to(prefix)


def test_environment_manifest_is_project_scoped():
    text = (ROOT / "environment.yml").read_text()
    assert "name: wdb-classifier-cpu" in text
    assert "- -e .[features,ztf,eclipse,dev]" in text
    assert all(value not in text for value in ("/home/", "/Users/", "prefix:", "system-site-packages"))


def test_repository_audit_rejects_forced_private_artifacts(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "model.joblib").write_bytes(b"not-a-real-model")
    spec = importlib.util.spec_from_file_location("repository_audit", ROOT / "scripts/audit_repository.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(tmp_path)
    assert not result["complete"] and result["artifact_findings"][0]["file"] == "model.joblib"
