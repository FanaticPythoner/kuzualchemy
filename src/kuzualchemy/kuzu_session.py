from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Type, TypeVar
from atp_pipeline import DbBulkAction
from .kuzu_connection import KuzuConnection
from .kuzu_orm import KuzuRelationshipBase
from .kuzu_relationship_read import relationship_read_direction, relationship_read_name, relationship_read_pairs
from .kuzu_session_rows import (
    RelationshipRoute,
    _node_delete_row,
    _node_label,
    _node_row,
    _pk_row,
    _primary_key_fields,
    _relationship_route,
    _relationship_row,
    _relationship_update_row,
)
ModelType = TypeVar("ModelType")
class KuzuSession:
    """Collect ORM objects and submit typed DB work to ATP."""
    def __init__(
        self,
        connection: KuzuConnection | None = None,
        db_path: str | Path | None = None,
        autoflush: bool = True,
        autocommit: bool = False,
        expire_on_commit: bool = True,
        bulk_insert_threshold: int = 10,
        bulk_batch_size: int = 10000,
        force_gc: bool = False,
        bulk_batch_size_max: int = 65536,
    ) -> None:
        if connection is None and db_path is None:
            raise ValueError("connection or db_path is required")
        self._conn = connection or KuzuConnection(db_path)  # type: ignore[arg-type]
        self._owns_connection = connection is None
        self.autoflush = autoflush
        self.autocommit = autocommit
        self.expire_on_commit = expire_on_commit
        self.bulk_insert_threshold = int(bulk_insert_threshold)
        self.bulk_batch_size = int(bulk_batch_size)
        self.bulk_batch_size_max = int(bulk_batch_size_max)
        self._new: list[Any] = []
        self._dirty: list[Any] = []
        self._deleted: list[Any] = []
        self._identity_map: dict[str, Any] = {}
    def get_db_path(self) -> str:
        return str(self._conn.db_path)
    @property
    def connection(self) -> KuzuConnection:
        return self._conn
    def query(self, model_class: Type[ModelType], alias: str = "n"):
        from .kuzu_query import Query
        return Query(model_class, session=self, alias=alias)
    def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        as_iterator: bool = False,
        page_size: int | None = None,
        prefetch_pages: int = 1,
    ) -> list[dict[str, Any]] | Iterator[dict[str, Any]]:
        self._flush_for_read()
        rows = self._conn.execute(query, parameters or {})
        return iter(rows) if as_iterator else rows
    def _execute_for_query_object(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.execute(query, parameters or {})
    def _execute_relationship_read_for_query_object(
        self,
        relationship_class: Type[Any],
        alias: str,
        pairs_subset: list[int] | None,
    ) -> list[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.read_relationships(
            relationship=relationship_read_name(relationship_class),
            alias=alias,
            direction=relationship_read_direction(relationship_class),
            pairs=relationship_read_pairs(relationship_class),
            pairs_subset=list(pairs_subset or []),
        )
    def _execute_many_for_query_object(self, queries: list[tuple[str, dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        self._flush_for_read()
        return self._conn.execute_many(queries)
    def iterate(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        page_size: int = 1000,
        prefetch_pages: int = 1,
    ) -> Iterator[dict[str, Any]]:
        return iter(self.execute(query, parameters))
    def add(self, instance: Any) -> None:
        if instance not in self._new:
            self._new.append(instance)
        self._remember(instance)
        if self.autocommit:
            self.commit()
    def add_all(self, instances: list[Any], batch_size: int | None = None) -> None:
        for instance in instances:
            self.add(instance)
    def create_relationship(self, relationship_class: Type[Any], from_node: Any, to_node: Any, **properties: Any) -> Any:
        if not issubclass(relationship_class, KuzuRelationshipBase):
            raise ValueError(f"{relationship_class.__name__} must inherit from KuzuRelationshipBase")
        relationship = relationship_class.create_between(from_node, to_node, **properties)
        self.add(relationship)
        return relationship
    def delete(self, instance: Any) -> None:
        if instance in self._new:
            self._new.remove(instance)
        if instance not in self._deleted:
            self._deleted.append(instance)
        if self.autocommit:
            self.commit()
    def merge(self, instance: Any) -> Any:
        current = self._identity_map.get(self._identity_key(instance))
        if current is None:
            self.add(instance)
            return instance
        for name, value in instance.model_dump(mode="python").items():
            setattr(current, name, value)
        if current not in self._dirty:
            self._dirty.append(current)
        return current
    def bulk_insert_immediate(self, instances: list[Any], batch_size: int | None = None) -> None:
        self._write_instances(DbBulkAction.CREATE, instances)
    def bulk_insert_graph_immediate(
        self,
        node_instances: list[Any],
        relationship_instances: list[Any],
    ) -> None:
        self._write_instance_groups(
            DbBulkAction.CREATE,
            node_instances,
            relationship_instances,
        )
    def bulk_update_nodes(self, model_class: Type[Any], rows: list[dict[str, Any]]) -> None:
        self._conn.bulk_write_nodes(DbBulkAction.UPDATE, _node_label(model_class), rows, _primary_key_fields(model_class))
    def bulk_update_node_groups(self, rows_by_class: dict[Type[Any], list[dict[str, Any]]]) -> None:
        self._conn.bulk_write_nodes_many(
            [
                (DbBulkAction.UPDATE, _node_label(cls), rows, _primary_key_fields(cls))
                for cls, rows in rows_by_class.items()
            ]
        )
    def bulk_delete_nodes(self, model_class: Type[Any], pks: list[Any]) -> None:
        key_fields = _primary_key_fields(model_class)
        self._conn.bulk_write_nodes(DbBulkAction.DELETE, _node_label(model_class), [_pk_row(key_fields, pk) for pk in pks], key_fields)
    def bulk_update_relationships(self, instances: list[Any], fields: list[str]) -> None:
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        for instance in instances:
            cls = type(instance)
            if not hasattr(cls, "__kuzu_rel_name__"):
                raise TypeError(f"{cls.__name__} is not a registered Kuzu relationship")
            row = _relationship_update_row(instance, fields)
            rels.setdefault(_relationship_route(cls, row), []).append(row)
        self._conn.bulk_write_relationships_many(
            [
                (DbBulkAction.UPDATE, rel_type, from_label, to_label, rows, [from_key], [to_key])
                for (rel_type, from_label, to_label, from_key, to_key), rows in rels.items()
            ]
        )
    def flush(self) -> None:
        self._write_instances(DbBulkAction.CREATE, self._new)
        self._write_instances(DbBulkAction.UPDATE, self._dirty)
        self._write_instances(DbBulkAction.DELETE, self._deleted)
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()
    def commit(self) -> None:
        self.flush()
        if self.expire_on_commit:
            self.expire_all()
    def rollback(self) -> None:
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()
        self._identity_map.clear()
    def expire(self, instance: Any) -> None:
        self._identity_map.pop(self._identity_key(instance), None)
    def expire_all(self) -> None:
        self._identity_map.clear()
    def close(self) -> None:
        self.rollback()
        if self._owns_connection:
            self._conn.close()
    @contextmanager
    def begin(self) -> Iterator["KuzuSession"]:
        try:
            yield self
            self.commit()
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError, LookupError):
            self.rollback()
            raise
    def _flush_for_read(self) -> None:
        if self.autoflush and (self._new or self._dirty or self._deleted):
            self.flush()
    def _remember(self, instance: Any) -> None:
        if hasattr(type(instance), "__kuzu_node_name__"):
            self._identity_map[self._identity_key(instance)] = instance
    def _identity_key(self, instance: Any) -> str:
        cls = type(instance)
        values = tuple(getattr(instance, field) for field in _primary_key_fields(cls))
        return f"{cls.__module__}.{cls.__qualname__}:{values!r}"
    def _write_instances(self, action: DbBulkAction, instances: list[Any]) -> None:
        nodes: dict[type[Any], list[dict[str, Any]]] = {}
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        for instance in instances:
            cls = type(instance)
            if hasattr(cls, "__kuzu_node_name__"):
                rows = _node_delete_row(instance) if action == DbBulkAction.DELETE else _node_row(instance)
                nodes.setdefault(cls, []).append(rows)
                if action == DbBulkAction.DELETE:
                    self.expire(instance)
                else:
                    self._remember(instance)
            elif hasattr(cls, "__kuzu_rel_name__"):
                row = _relationship_row(instance)
                rels.setdefault(_relationship_route(cls, row), []).append(row)
            else:
                raise TypeError(f"{cls.__name__} is not a registered Kuzu model")
        self._write_grouped_rows(action, nodes, rels)
    def _write_instance_groups(
        self,
        action: DbBulkAction,
        node_instances: list[Any],
        relationship_instances: list[Any],
    ) -> None:
        nodes: dict[type[Any], list[dict[str, Any]]] = {}
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        for instance in node_instances:
            cls = type(instance)
            if not hasattr(cls, "__kuzu_node_name__"):
                raise TypeError(f"{cls.__name__} is not a registered Kuzu node")
            row = _node_delete_row(instance) if action == DbBulkAction.DELETE else _node_row(instance)
            nodes.setdefault(cls, []).append(row)
            if action == DbBulkAction.DELETE:
                self.expire(instance)
            else:
                self._remember(instance)
        for instance in relationship_instances:
            cls = type(instance)
            if not hasattr(cls, "__kuzu_rel_name__"):
                raise TypeError(f"{cls.__name__} is not a registered Kuzu relationship")
            row = _relationship_row(instance)
            rels.setdefault(_relationship_route(cls, row), []).append(row)
        self._write_grouped_rows(action, nodes, rels)
    def _write_grouped_rows(
        self,
        action: DbBulkAction,
        nodes: dict[type[Any], list[dict[str, Any]]],
        rels: dict[RelationshipRoute, list[dict[str, Any]]],
    ) -> None:
        self._conn.bulk_write_nodes_and_relationships_many(
            [(action, _node_label(cls), rows, _primary_key_fields(cls)) for cls, rows in nodes.items()],
            [
                (action, rel_type, from_label, to_label, rows, [from_key], [to_key])
                for (rel_type, from_label, to_label, from_key, to_key), rows in rels.items()
            ],
        )
