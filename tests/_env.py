"""Test environment loader for Kuzualchemy pytest suite.

This module ensures that the repository-level `.env` file is loaded before any
tests run. All parsing and environment application is implemented in Rust in
the `atp_core::env` module; here we simply delegate to the public
``atp_pipeline.load_workspace_dotenv`` helper so Rust-only and Python-driven
entrypoints share identical semantics.
"""

from __future__ import annotations

import os

from atp_pipeline import load_workspace_dotenv


load_workspace_dotenv(required=True)

os.environ["ATP_PROFILE"] = "0"
os.environ.pop("ATP_PROFILE_FREQ", None)
os.environ.pop("ATP_PROFILE_FLAMEGRAPH", None)
os.environ.pop("ATP_PROFILE_SPEEDSCOPE", None)

