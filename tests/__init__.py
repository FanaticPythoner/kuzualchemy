# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for KuzuAlchemy ORM.

This package contains tests for all components of the KuzuAlchemy ORM:
- Unit tests for individual components
- Integration tests for end-to-end functionality
- Performance benchmarks
- Edge case testing
"""

from pathlib import Path
import tempfile

# Load dotenv/.env configuration for tests before importing other modules
from . import _env  # noqa: F401  # pylint: disable=unused-import

TEST_DB_PATH = Path(tempfile.gettempdir()) / "test_kuzu_db"
BENCHMARK_DB_PATH = Path(tempfile.gettempdir()) / "benchmark_kuzu_db"


def cleanup_test_db(db_path: Path) -> None:
    if db_path.exists():
        import shutil

        shutil.rmtree(db_path, ignore_errors=True)


def create_test_db_path() -> Path:
    import uuid

    return Path(tempfile.gettempdir()) / f"test_kuzu_{uuid.uuid4().hex[:8]}"


__all__ = [
    "TEST_DB_PATH",
    "BENCHMARK_DB_PATH",
    "cleanup_test_db",
    "create_test_db_path",
]