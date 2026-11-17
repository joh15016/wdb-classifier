"""Credential-aware ZTF retrieval without modifying the scientific entry points."""

import argparse
import json
import os
import resource
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wdb-ztf-fetch")
    for name in ("ids", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-new-sources", type=int, default=20)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument(
        "--credentials-file", type=Path, help="External YAML file with matching hosts.<server> credentials"
    )
    choices.add_argument(
        "--password-login", action="store_true", help="Use WDB_ZTF_USERNAME and WDB_ZTF_PASSWORD"
    )
    choices.add_argument("--token-env", help="Token variable name; default WDB_ZTF_TOKEN")
    args = parser.parse_args(argv)

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4 * 1024**3
    if soft == resource.RLIM_INFINITY or soft > cap:
        resource.setrlimit(
            resource.RLIMIT_AS, (cap if hard == resource.RLIM_INFINITY else min(cap, hard), hard)
        )

    from ..ztf import ZTFConfig
    from . import Authentication, fetch_authenticated

    authentication = Authentication(
        credentials_file=args.credentials_file,
        password_login=args.password_login,
        token_env=args.token_env or "WDB_ZTF_TOKEN",
    )
    try:
        result = fetch_authenticated(
            args.ids,
            ZTFConfig.load(args.config),
            args.output,
            authentication=authentication,
            max_new_sources=args.max_new_sources,
        )
    except (ValueError, RuntimeError, ImportError):
        parser.exit(
            2, "ZTF retrieval failed; check the selected login, catalog, connection and checkpoint inputs.\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
