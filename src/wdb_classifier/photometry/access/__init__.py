"""Explicit authentication for the existing, checkpointed ZTF fetcher.

The adapter keeps credentials out of data and manifests. Completed downloads
can be reopened without loading credentials or contacting the server.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Authentication:
    """Credential locations only; never put actual secrets in this object."""

    credentials_file: Path | None = None
    password_login: bool = False
    token_env: str = "WDB_ZTF_TOKEN"
    username_env: str = "WDB_ZTF_USERNAME"
    password_env: str = "WDB_ZTF_PASSWORD"

    def __post_init__(self):
        if self.credentials_file is not None and self.password_login:
            raise ValueError("Choose a credential file or environment-based password login, not both")
        if any(not name.isidentifier() for name in (self.token_env, self.username_env, self.password_env)):
            raise ValueError("Credential environment-variable names must be valid identifiers")

    def connection_kwargs(self, host):
        if self.credentials_file is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError("Install the project's ztf extra for credential-file support") from None
            key = host.removesuffix(".caltech.edu")
            try:
                document = yaml.safe_load(Path(self.credentials_file).expanduser().read_text())
                entry = document["hosts"][key]
                username, password = entry["username"], entry["password"]
            except Exception:
                # YAML errors can include the offending line, which may be a secret.
                raise ValueError(
                    "Cannot read matching hosts.<server>.username/password credentials from the selected file"
                ) from None
        elif self.password_login:
            username, password = os.environ.get(self.username_env), os.environ.get(self.password_env)
        else:
            token = os.environ.get(self.token_env, "").strip()
            if not token:
                raise ValueError(
                    "Set the selected token environment variable or explicitly choose password login"
                )
            return {"token": token}
        if not all(isinstance(value, str) and value.strip() for value in (username, password)):
            raise ValueError("The selected login requires a nonempty username and password")
        return {"username": username.strip(), "password": password}


def _close(client):
    if client is not None:
        try:
            client.close_all()
        except Exception:
            pass


def _connect(config, authentication):
    from penquins import Kowalski

    credentials = authentication.connection_kwargs(config.host)
    client = None
    try:
        client = Kowalski(
            timeout=config.timeout,
            verbose=False,
            max_retries=0,
            instances={
                "kowalski": {
                    "protocol": "https",
                    "host": config.host,
                    "port": config.port,
                    **credentials,
                }
            },
        )
        if not client.instance_has_catalog(config.collection, name="kowalski"):
            raise RuntimeError("Catalog not accessible")
        return client
    except Exception:
        _close(client)
        # Penquins exceptions can include server replies; never propagate them.
        raise RuntimeError(
            "Kowalski login/catalog access failed; check the selected credentials, host and collection"
        ) from None


class _LazyClient:
    """Do not load credentials or contact the server when all IDs are cached."""

    def __init__(self, config, authentication):
        self.config, self.authentication = config, authentication
        self.client = None
        self.failure = None

    def query(self, **kwargs):
        if self.failure is not None:
            raise RuntimeError(self.failure) from None
        if self.client is None:
            try:
                self.client = _connect(self.config, self.authentication)
            except (ValueError, RuntimeError, ImportError) as error:
                # The ordinary query retry loop must not repeatedly attempt a bad login.
                self.failure = str(error)
                raise RuntimeError(self.failure) from None
        return self.client.query(**kwargs)


def fetch_authenticated(ids_path, config, output, *, authentication=None, max_new_sources=20):
    """Use the ordinary query, cleaning, manifests and per-source checkpoints."""
    from ..ztf import fetch

    client = _LazyClient(config, authentication or Authentication())
    try:
        return fetch(ids_path, config, output, max_new_sources=max_new_sources, client=client)
    except RuntimeError:
        if client.failure is not None:
            raise RuntimeError(client.failure) from None
        raise
    finally:
        _close(client.client)
