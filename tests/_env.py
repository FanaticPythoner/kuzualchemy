"""Test environment loader for Kuzualchemy pytest suite.

This module loads a dedicated test environment file (`tests/.env.test`) before
any tests run. Keys in the file are intentionally namespaced to avoid
cross-project collisions when other repositories expose similarly named runtime
variables.
"""

from __future__ import annotations

import os
from pathlib import Path


_TEST_ENV_PATH = Path(__file__).with_name(".env.test")

_TEST_TO_RUNTIME_ENV_KEY_MAP = {
    "KUZUALCHEMY_TEST_OMP_NUM_THREADS": "OMP_NUM_THREADS",
    "KUZUALCHEMY_TEST_MKL_NUM_THREADS": "MKL_NUM_THREADS",
    "KUZUALCHEMY_TEST_NUMBA_NUM_THREADS": "NUMBA_NUM_THREADS",
    "KUZUALCHEMY_TEST_ATP_PROFILE": "ATP_PROFILE",
    "KUZUALCHEMY_TEST_ATP_READONLY_POOL_MAX_SIZE": "ATP_READONLY_POOL_MAX_SIZE",
    "KUZUALCHEMY_TEST_ATP_READONLY_POOL_WARM_COUNT": "ATP_READONLY_POOL_WARM_COUNT",
    "KUZUALCHEMY_TEST_ATP_READONLY_BUFFER_POOL_BYTES": "ATP_READONLY_BUFFER_POOL_BYTES",
}


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"Missing required test env file: {path}")

    parsed: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid env entry in {path} at line {line_no}: expected KEY=VALUE"
            )
        key_part, value_part = line.split("=", 1)
        key = key_part.strip()
        value = value_part.strip()
        if not key:
            raise ValueError(
                f"Invalid env entry in {path} at line {line_no}: empty KEY"
            )
        if key in parsed:
            raise ValueError(
                f"Duplicate env key in {path} at line {line_no}: {key}"
            )
        parsed[key] = value
    return parsed


def _apply_runtime_test_env(parsed_env: dict[str, str]) -> None:
    missing_required_keys = [
        key for key in _TEST_TO_RUNTIME_ENV_KEY_MAP if key not in parsed_env
    ]
    if missing_required_keys:
        missing_csv = ", ".join(sorted(missing_required_keys))
        raise RuntimeError(
            f"Missing required keys in {_TEST_ENV_PATH}: {missing_csv}"
        )

    for source_key, target_key in _TEST_TO_RUNTIME_ENV_KEY_MAP.items():
        os.environ[target_key] = parsed_env[source_key]

    os.environ.pop("ATP_PROFILE_FREQ", None)
    os.environ.pop("ATP_PROFILE_FLAMEGRAPH", None)
    os.environ.pop("ATP_PROFILE_SPEEDSCOPE", None)


_apply_runtime_test_env(_parse_env_file(_TEST_ENV_PATH))

