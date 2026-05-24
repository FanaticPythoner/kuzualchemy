from __future__ import annotations

from typing import Iterable

from atp_pipeline import split_cypher_statements

from .kuzu_connection import KuzuConnection
from .kuzu_orm import get_all_ddl
from .kuzu_session import KuzuSession


class DDLExecutor:
    """Submit DDL statements through a KuzuConnection."""

    def __init__(self, connection: KuzuConnection) -> None:
        self.connection = connection

    def execute_all_ddl(self, ddl: str | None = None) -> None:
        statements = split_ddl(ddl if ddl is not None else get_all_ddl())
        if statements:
            self.connection.schema_apply(statements)


def split_ddl(ddl: str) -> list[str]:
    return split_cypher_statements(ddl)


def connection_for_schema(target: KuzuConnection | KuzuSession) -> KuzuConnection:
    if isinstance(target, KuzuConnection):
        return target
    if isinstance(target, KuzuSession):
        return target.connection
    raise TypeError("target must be KuzuConnection or KuzuSession")


def initialize_schema(target: KuzuConnection | KuzuSession, ddl: str | None = None) -> None:
    DDLExecutor(connection_for_schema(target)).execute_all_ddl(ddl)


def execute_ddl(target: KuzuConnection | KuzuSession, statement: str) -> None:
    connection_for_schema(target).schema_apply([statement])


def execute_many_ddl(target: KuzuConnection | KuzuSession, statements: Iterable[str]) -> None:
    connection_for_schema(target).schema_apply([statement for statement in statements if statement.strip()])
