import json
import sys
from types import SimpleNamespace

import pytest

from wdb_classifier.photometry.access import Authentication, fetch_authenticated
from wdb_classifier.photometry.ztf import ZTFConfig


def fake_backend(monkeypatch, *, login_error=None, has_catalog=True):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if login_error:
                raise RuntimeError(login_error)
            self.closed = False

        def instance_has_catalog(self, name_of_catalog, *, name):
            assert name_of_catalog == "ZTF_sources_20240515" and name == "kowalski"
            return has_catalog

        def query(self, *, queries, use_batch_query, max_n_threads):
            assert use_batch_query and max_n_threads == 1
            ids = queries[0]["query"]["filter"]["_id"]["$in"]
            return {
                "kowalski": [
                    {
                        "status": "success",
                        "data": [
                            dict(
                                _id=sid,
                                filter=1,
                                ra=10.0,
                                dec=20.0,
                                data=[dict(hjd=2450000.0, mag=18.0, magerr=0.1, catflags=0, programid=1)],
                            )
                            for sid in ids
                        ],
                    }
                ]
            }

        def close_all(self):
            self.closed = True
            calls.append("closed")

    monkeypatch.setitem(sys.modules, "penquins", SimpleNamespace(Kowalski=Client))
    from wdb_classifier.photometry import ztf

    monkeypatch.setattr(ztf.time, "sleep", lambda _: None)
    return calls


def inputs(tmp_path):
    ids = tmp_path / "ids.csv"
    ids.write_text("ztf_id\n11\n")
    return ids, ZTFConfig("gloria.caltech.edu", "ZTF_sources_20240515")


def test_external_login_overrides_token_and_reuses_checkpoint(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    monkeypatch.setenv("WDB_ZTF_TOKEN", "invalid-old-token")
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(  # pragma: allowlist secret -- deliberately fictitious test credentials
        "hosts:\n  gloria:\n    username: sample-user\n    password: sample-secret\n"  # pragma: allowlist secret
    )
    ids, config = inputs(tmp_path)
    output = tmp_path / "output"
    auth = Authentication(credentials_file=credentials)
    result = fetch_authenticated(ids, config, output, authentication=auth, max_new_sources=1)
    assert result["complete"] and result["sources_with_photometry"] == 1
    params = calls[0]
    assert params["verbose"] is False and params["max_retries"] == 0
    assert params["instances"]["kowalski"]["username"] == "sample-user"
    assert params["instances"]["kowalski"]["password"] == "sample-secret"  # pragma: allowlist secret
    assert "token" not in params["instances"]["kowalski"] and calls[1] == "closed"
    checkpoint = output / "sources/11.json"
    before = checkpoint.stat().st_mtime_ns
    credentials.unlink()
    assert fetch_authenticated(ids, config, output, authentication=auth)["new_sources"] == 0
    assert checkpoint.stat().st_mtime_ns == before and len(calls) == 2
    for path in output.rglob("*.json"):
        text = path.read_text()
        assert all(secret not in text for secret in ("sample-user", "sample-secret", "invalid-old-token"))


@pytest.mark.parametrize("password_login", [False, True])
def test_environment_credentials(tmp_path, monkeypatch, password_login):
    calls = fake_backend(monkeypatch)
    monkeypatch.setenv("WDB_ZTF_TOKEN", "sample-token")
    monkeypatch.setenv("WDB_ZTF_USERNAME", "sample-user")
    monkeypatch.setenv("WDB_ZTF_PASSWORD", "sample-secret")
    ids, config = inputs(tmp_path)
    assert fetch_authenticated(
        ids, config, tmp_path / "out", authentication=Authentication(password_login=password_login)
    )["complete"]
    credentials = calls[0]["instances"]["kowalski"]
    if password_login:
        assert credentials["username"] == "sample-user" and "token" not in credentials
    else:
        assert credentials["token"] == "sample-token" and "username" not in credentials


def test_failed_login_is_sanitized_not_retried_or_cached(tmp_path, monkeypatch, capsys):
    calls = fake_backend(monkeypatch, login_error="SERVER ECHOED sample-secret sample-token")
    monkeypatch.setenv("WDB_ZTF_TOKEN", "sample-token")
    ids, config = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="login/catalog access failed") as error:
        fetch_authenticated(ids, config, tmp_path / "out")
    assert len(calls) == 1
    assert not list((tmp_path / "out/sources").glob("*.json"))
    assert "sample-secret" not in str(error.value) and "sample-token" not in str(error.value)
    assert "sample-secret" not in capsys.readouterr().out


def test_inaccessible_catalog_is_not_recorded_as_no_match(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch, has_catalog=False)
    monkeypatch.setenv("WDB_ZTF_TOKEN", "sample-token")
    ids, config = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="login/catalog access failed"):
        fetch_authenticated(ids, config, tmp_path / "out")
    assert calls[-1] == "closed" and len(calls) == 2
    assert not list((tmp_path / "out/sources").glob("*.json"))


def test_only_matching_host_entry_and_sanitized_yaml_errors(tmp_path):
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text("hosts:\n  melman:\n    username: sample-user\n    password: sample-secret\n")
    auth = Authentication(credentials_file=credentials)
    with pytest.raises(ValueError, match="matching hosts"):
        auth.connection_kwargs("gloria.caltech.edu")
    credentials.write_text("hosts: [ sample-secret\n")
    with pytest.raises(ValueError) as error:
        auth.connection_kwargs("gloria.caltech.edu")
    assert "sample-secret" not in str(error.value)


def test_zero_budget_needs_no_credentials_or_connection(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    monkeypatch.delenv("WDB_ZTF_TOKEN", raising=False)
    ids, config = inputs(tmp_path)
    output = tmp_path / "out"
    result = fetch_authenticated(ids, config, output, max_new_sources=0)
    assert not result["complete"] and result["new_sources"] == 0 and calls == []
    assert json.loads((output / "status.json").read_text()) == result
