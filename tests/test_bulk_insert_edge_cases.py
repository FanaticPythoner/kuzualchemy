# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
import tempfile
from datetime import datetime, date, timedelta
from typing import List, Optional

import pytest

from kuzualchemy import (
    KuzuBaseModel,
    kuzu_node,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
    get_ddl_for_node,
)
from kuzualchemy.constants import KuzuDefaultFunction
from kuzualchemy.test_utilities import initialize_schema


def _new_session(tmp_dir: Optional[str] = None, **session_kwargs) -> KuzuSession:
    """Create a new session backed by a fresh temp directory."""
    workdir = tmp_dir or tempfile.mkdtemp(prefix="kuzu_bulk_edge_")
    db_path = os.path.join(workdir, "db")
    return KuzuSession(db_path=db_path, **session_kwargs)


@kuzu_node("AllTypesNode")
class AllTypesNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    small_i8: int = kuzu_field(kuzu_type=KuzuDataType.INT8)
    small_i16: int = kuzu_field(kuzu_type=KuzuDataType.INT16)
    mid_i32: int = kuzu_field(kuzu_type=KuzuDataType.INT32)
    big_i64: int = kuzu_field(kuzu_type=KuzuDataType.INT64)
    huge_i128: int = kuzu_field(kuzu_type=KuzuDataType.INT128)
    dec: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)
    flag: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL)
    when_ts: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    the_date: date = kuzu_field(kuzu_type=KuzuDataType.DATE)
    u: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID)
    ints: List[int] = kuzu_field(kuzu_type="INT64[]")
    strs: List[str] = kuzu_field(kuzu_type="STRING[]")


def test_massive_bulk_insert_all_types_and_arrays_fast():
    """Massive bulk insert with diverse types and arrays; validates round-trip."""
    session = _new_session(bulk_insert_threshold=1000)
    ddl = get_ddl_for_node(AllTypesNode)
    initialize_schema(session, ddl=ddl)

    N = 8000  # large but CI-friendly
    now = datetime.now()
    base_date = date(2000, 1, 1)

    rows: List[AllTypesNode] = []
    for i in range(N):
        # Cover extreme integer values periodically
        i8 = (-128 if i % 97 == 0 else 127 if i % 89 == 0 else (i % 127) - 64)
        i16 = (-32768 if i % 193 == 0 else 32767 if i % 181 == 0 else (i % 10000) - 5000)
        i32 = (-2_147_483_648 if i % 389 == 0 else 2_147_483_647 if i % 383 == 0 else i)
        i64 = (-9_223_372_036_854_775_808 if i % 997 == 0 else 9_223_372_036_854_775_807 if i % 991 == 0 else i * 10)
        # Keep i128 within 64-bit range to avoid Arrow Python int conversion overflow
        i128 = i64
        dec = (i % 1000) / 7.0
        flag = (i % 3 == 0)
        when_ts = now - timedelta(seconds=i % 100000)
        the_date = base_date + timedelta(days=i % 20000)
        uid = uuid.uuid4()
        ints = [i % 2, (i // 2) % 2, (i // 3) % 2]
        strs = [f"s{i%10}", "αβγ", "emoji_😀"]

        rows.append(AllTypesNode(
            id=i + 1,
            small_i8=i8,
            small_i16=i16,
            mid_i32=i32,
            big_i64=i64,
            huge_i128=i128,
            dec=dec,
            flag=flag,
            when_ts=when_ts,
            the_date=the_date,
            u=uid,
            ints=ints,
            strs=strs,
        ))

    start = time.time()
    session._bulk_insert(rows)
    elapsed = time.time() - start

    # Validate count
    res = session.execute("MATCH (n:AllTypesNode) RETURN count(n) AS cnt")
    cnt = res[0].get("cnt", list(res[0].values())[0])
    assert cnt == N

    # Validate a few edge rows round-trip for arrays and types
    sample = list(session.execute(
        "MATCH (n:AllTypesNode) WHERE n.id IN [1, 97, 181, 383, 997] RETURN n"
    ))
    assert sample, "No rows returned in sample"
    for row in sample:
        node = row.get("n") if isinstance(row, dict) else list(row.values())[0]
        assert isinstance(node["u"], uuid.UUID)
        assert isinstance(node["when_ts"], datetime)
        assert isinstance(node["the_date"], date)
        assert isinstance(node["ints"], list)
        assert isinstance(node["strs"], list)

    # Keep the test assertive but avoid strict timing (env-dependent)
    assert elapsed < 10.0, f"Insert too slow: {elapsed:.2f}s for {N} rows"


@kuzu_node("UnicodeNode")
class UnicodeNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    text: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    tags: List[str] = kuzu_field(kuzu_type="STRING[]")


def test_bulk_insert_unicode_and_large_strings():
    """Bulk insert with large/unicode strings and verify integrity."""
    session = _new_session(bulk_insert_threshold=1000)
    ddl = get_ddl_for_node(UnicodeNode)
    initialize_schema(session, ddl=ddl)

    N = 5000
    long = "x" * 5000
    emoji = "😀🐍🚀✨🔥"

    rows = [
        UnicodeNode(
            id=i + 1,
            text=(f"Row_{i}-" + long + "-" + emoji),
            tags=[f"t{i%7}", emoji, "αβγ", long[:1000]],
        )
        for i in range(N)
    ]

    session._bulk_insert(rows)

    # Validate subset and special characters intact
    out = list(session.execute("MATCH (n:UnicodeNode) WHERE n.id IN [1, 2500, 5000] RETURN n"))
    for row in out:
        node = row.get("n") if isinstance(row, dict) else list(row.values())[0]
        assert emoji in node["text"]
        assert any(emoji in t for t in node["tags"])  # emoji preserved in arrays


@kuzu_node("DefaultsNode")
class DefaultsNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=KuzuDefaultFunction.CURRENT_TIMESTAMP)
    d: Optional[date] = kuzu_field(kuzu_type=KuzuDataType.DATE, default=None)


def test_bulk_insert_default_functions_resolution():
    """Ensure default functions are materialized during bulk insert when fields omitted."""
    session = _new_session(bulk_insert_threshold=500)
    ddl = get_ddl_for_node(DefaultsNode)
    initialize_schema(session, ddl=ddl)

    N = 2000
    rows = [DefaultsNode(id=i+1) for i in range(N)]  # omit created_at explicitly; id provided

    session._bulk_insert(rows)

    # All rows should be present
    res = session.execute("MATCH (n:DefaultsNode) RETURN count(n) AS c")
    c = res[0].get("c", list(res[0].values())[0])
    assert c == N

    # Sample a few to ensure defaults are non-null and typed
    sample = list(session.execute("MATCH (n:DefaultsNode) RETURN n LIMIT 5"))
    for row in sample:
        node = row.get("n") if isinstance(row, dict) else list(row.values())[0]
        assert isinstance(node["id"], int)
        assert isinstance(node["created_at"], datetime)


@kuzu_node("PKDupNode")
class PKDupNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


def test_bulk_insert_within_batch_duplicate_primary_keys_rolls_back():
    """Single bulk insert with duplicated PKs in-batch must fail and leave table empty."""
    session = _new_session(bulk_insert_threshold=100)
    ddl = get_ddl_for_node(PKDupNode)
    initialize_schema(session, ddl=ddl)

    # Create a batch with intentional duplicate IDs
    rows = [PKDupNode(id=i, name=f"r{i}") for i in range(1, 401)]
    rows += [PKDupNode(id=i, name=f"dup{i}") for i in range(1, 51)]  # 50 duplicates

    with pytest.raises(RuntimeError) as exc:
        session._bulk_insert(rows)
    assert any(k in str(exc.value).lower() for k in ("primary", "duplicate", "constraint"))

    # Table should still be empty
    res = session.execute("MATCH (n:PKDupNode) RETURN count(n) AS c")
    c = res[0].get("c", list(res[0].values())[0])
    assert c == 0


@kuzu_node("PKNode")
class PKNode(KuzuBaseModel):
    id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


def test_bulk_insert_duplicate_primary_keys_across_batches_rollback():
    """Second bulk insert containing duplicate PKs must fail and not change counts."""
    session = _new_session(bulk_insert_threshold=200)
    ddl = get_ddl_for_node(PKNode)
    initialize_schema(session, ddl=ddl)

    # Batch 1: all unique
    batch1 = [PKNode(id=uuid.uuid4(), name=f"r{i}") for i in range(500)]
    session._bulk_insert(batch1)
    res = session.execute("MATCH (n:PKNode) RETURN count(n) AS c1")
    before = res[0].get("c1", list(res[0].values())[0])
    assert before == 500

    # Batch 2: include duplicates from batch1
    dup_ids = [batch1[i].id for i in range(0, 500, 25)]  # 20 duplicates
    batch2 = [
        PKNode(id=uuid.uuid4(), name="fresh") for _ in range(480)
    ] + [
        PKNode(id=did, name="dupe") for did in dup_ids
    ]

    with pytest.raises(RuntimeError) as exc:
        session._bulk_insert(batch2)
    # Error message should mention constraint/primary/duplicate in a broad sense
    assert any(k in str(exc.value).lower() for k in ("constraint", "primary", "duplicate"))

    # Count unchanged
    res2 = session.execute("MATCH (n:PKNode) RETURN count(n) AS c2")
    after = res2[0].get("c2", list(res2[0].values())[0])
    assert after == before

