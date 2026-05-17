# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from atp_pipeline import OpKind, OperationSpec

from kuzualchemy import KuzuBaseModel, KuzuDataType, KuzuSession, kuzu_field, kuzu_node


@kuzu_node("BulkSubmitManyA")
class BulkSubmitManyA(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_node("BulkSubmitManyB")
class BulkSubmitManyB(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


class _CaptureATP:
    def __init__(self) -> None:
        self.submissions: list[list[OperationSpec]] = []

    def create_nodes_spec(
        self,
        label: str,
        rows: list[dict[str, Any]],
        *,
        return_rows: bool,
        pk_fields: list[str],
    ) -> OperationSpec:
        return OperationSpec(
            kind=OpKind.CREATE_NODE,
            label=label,
            props={"rows": rows, "return_rows": return_rows, "pk_fields": pk_fields},
            context={"label": label},
        )

    def submit_specs(self, specs: list[OperationSpec]) -> list[dict[str, Any]]:
        self.submissions.append(specs)
        return [{} for _ in specs]

    def _extract_post_cypher_rows(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return []


class _CaptureConnection:
    def __init__(self, atp: _CaptureATP) -> None:
        self._atp = atp


def _session_for_capture(atp: _CaptureATP) -> KuzuSession:
    session = KuzuSession.__new__(KuzuSession)
    session._conn = _CaptureConnection(atp)
    session._disable_bulk_pipeline = False
    session.bulk_batch_size = 1000
    session.bulk_batch_size_max = 1000
    session._metadata_cache = OrderedDict()
    session._metadata_cache_size = 64
    session._identity_map = {}
    return session


def test_bulk_insert_submits_label_batches_in_one_atp_call() -> None:
    atp = _CaptureATP()
    session = _session_for_capture(atp)

    session._bulk_insert(
        [
            BulkSubmitManyA(id=3, name="a3"),
            BulkSubmitManyB(id=2, name="b2"),
            BulkSubmitManyA(id=1, name="a1"),
            BulkSubmitManyB(id=4, name="b4"),
        ],
        batch_size=1000,
    )

    assert len(atp.submissions) == 1
    labels = [spec.label for spec in atp.submissions[0]]
    assert labels == ["BulkSubmitManyA", "BulkSubmitManyB"]
    a_rows = atp.submissions[0][0].props["rows"]
    b_rows = atp.submissions[0][1].props["rows"]
    assert [row["id"] for row in a_rows] == [1, 3]
    assert [row["id"] for row in b_rows] == [2, 4]
