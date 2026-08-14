from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Type, TypeVar
from atp_pipeline import DbBulkAction, DbStatement, validate_kuzu_manual_auto_increment_values
from .constants import PerformanceConstants
from .kuzu_connection import KuzuConnection
from .kuzu_orm import KuzuRelationshipBase
from .kuzu_relationship_read import relationship_read_direction, relationship_read_name, relationship_read_pairs
from .kuzu_session_rows import (
    RelationshipRoute,
    apply_generated_values,
    auto_generation_fields,
    _node_delete_row,
    _node_label,
    _node_row,
    _pk_row,
    _primary_key_fields,
    _relationship_route,
    _relationship_row,
    _relationship_update_row,
    model_field_specs,
    node_create_spec,
    node_merge_policies,
    normalize_model_row,
    relationship_create_spec,
    validate_explicit_null_primary_keys,
)
ModelType = TypeVar("ModelType")
BulkIdentityAction = tuple[str, Any]
BulkRowPartition = tuple[
    dict[type[Any], list[dict[str, Any]]],
    dict[RelationshipRoute, list[dict[str, Any]]],
    list[BulkIdentityAction],
]
class KuzuSession:
    def __init__(
        self,
        connection: KuzuConnection | None = None,
        db_path: str | Path | None = None,
        autoflush: bool = True,
        autocommit: bool = False,
        expire_on_commit: bool = True,
        bulk_insert_threshold: int = PerformanceConstants.BATCH_INSERT_SIZE,
        bulk_batch_size: int = PerformanceConstants.BATCH_INSERT_SIZE,
        bulk_batch_size_max: int = PerformanceConstants.BATCH_INSERT_SIZE * 64,
        identity_tracking: bool = True,
    ) -> None:
        if connection is None and db_path is None:
            raise ValueError("connection or db_path is required")
        self._conn = connection or KuzuConnection(db_path)  # type: ignore[arg-type]
        self._owns_connection = connection is None
        self.autoflush = autoflush
        self.autocommit = autocommit
        self.expire_on_commit = expire_on_commit
        self.bulk_insert_threshold = bulk_insert_threshold
        self.bulk_batch_size = bulk_batch_size
        self.bulk_batch_size_max = bulk_batch_size_max
        self.identity_tracking = identity_tracking
        self._new: list[Any] = []
        self._dirty: list[Any] = []
        self._deleted: list[Any] = []
        self._new_object_ids: set[int] = set()
        self._dirty_object_ids: set[int] = set()
        self._deleted_object_ids: set[int] = set()
        self._identity_map: dict[str, Any] = {}
        self._identity_keys_by_object_id: dict[int, str] = {}
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
        if as_iterator:
            return self._conn.iterate(
                query,
                parameters or {},
                page_size=10 if page_size is None else page_size,
                prefetch_pages=prefetch_pages,
            )
        return self._conn.execute(query, parameters or {})
    def _execute_for_query_object(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.execute(query, parameters or {})
    def _execute_relationship_read_for_query_object(
        self,
        relationship_class: Type[Any],
        alias: str,
        pairs_subset: list[int] | None,
        filters: list[DbStatement] | None = None,
    ) -> list[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.read_relationships(
            relationship=relationship_read_name(relationship_class),
            alias=alias,
            direction=relationship_read_direction(relationship_class),
            pairs=relationship_read_pairs(relationship_class),
            pairs_subset=list(pairs_subset or []),
            filters=filters,
        )

    def _iterate_relationship_read_for_query_object(
        self,
        relationship_class: Type[Any],
        alias: str,
        pairs_subset: list[int] | None,
        filters: list[DbStatement] | None,
        page_size: int,
        prefetch_pages: int,
    ) -> Iterator[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.iterate_relationships(
            relationship=relationship_read_name(relationship_class),
            alias=alias,
            direction=relationship_read_direction(relationship_class),
            pairs=relationship_read_pairs(relationship_class),
            pairs_subset=list(pairs_subset or []),
            filters=filters,
            page_size=page_size,
            prefetch_pages=prefetch_pages,
        )
    def _iterate_for_query_object(
        self,
        query: str,
        parameters: dict[str, Any] | None,
        page_size: int,
        prefetch_pages: int,
        total_rows: int,
    ) -> Iterator[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.iterate(
            query,
            parameters or {},
            page_size=page_size,
            prefetch_pages=prefetch_pages,
            total_rows=total_rows,
        )
    def iterate(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        page_size: int = 1000,
        prefetch_pages: int = 1,
    ) -> Iterator[dict[str, Any]]:
        self._flush_for_read()
        return self._conn.iterate(
            query,
            parameters or {},
            page_size=page_size,
            prefetch_pages=prefetch_pages,
        )
    def add(self, instance: Any) -> None:
        self._add_pending(instance)
        if self.autocommit:
            self.commit()
    def _add_pending(self, instance: Any) -> None:
        self._append_pending(self._new, self._new_ids(), instance)
        self._remember(instance)
    def add_all(self, instances: list[Any]) -> None:
        for instance in instances:
            self._add_pending(instance)
        if self.autocommit:
            self.commit()
    def create_relationship(self, relationship_class: Type[Any], from_node: Any, to_node: Any, **properties: Any) -> Any:
        if not issubclass(relationship_class, KuzuRelationshipBase):
            raise ValueError(f"{relationship_class.__name__} must inherit from KuzuRelationshipBase")
        relationship = relationship_class.create_between(from_node, to_node, **properties)
        self.add(relationship)
        return relationship
    def delete(self, instance: Any) -> None:
        self._remove_pending(self._new, self._new_ids(), instance)
        self._append_pending(self._deleted, self._deleted_ids(), instance)
        if self.autocommit:
            self.commit()
    def merge(self, instance: Any) -> Any:
        current = self._identity_map.get(self._identity_key(instance))
        if current is None:
            self.add(instance)
            return instance
        for name, value in instance.model_dump(mode="python").items():
            setattr(current, name, value)
        self._append_pending(self._dirty, self._dirty_ids(), current)
        return current
    def bulk_insert_immediate(
        self,
        instances: list[Any],
        batch_size: int | None = None,
        *,
        track_identity: bool = False,
    ) -> None:
        self._write_instances(
            DbBulkAction.INSERT,
            instances,
            batch_size=batch_size,
            track_identity=track_identity,
        )
    def bulk_insert_graph_immediate(
        self,
        node_instances: list[Any],
        relationship_instances: list[Any],
        *,
        track_identity: bool = False,
    ) -> None:
        self._write_instance_groups(
            DbBulkAction.INSERT,
            node_instances,
            relationship_instances,
            track_identity=track_identity,
        )
    def bulk_update_nodes(self, model_class: Type[Any], rows: list[dict[str, Any]]) -> None:
        self._conn.bulk_write_nodes(
            DbBulkAction.UPDATE,
            _node_label(model_class),
            [normalize_model_row(model_class, dict(row)) for row in rows],
            _primary_key_fields(model_class),
            node_merge_policies(model_class),
        )
    def bulk_update_node_groups(self, rows_by_class: dict[Type[Any], list[dict[str, Any]]]) -> None:
        self._conn.bulk_write_nodes_many(
            [
                (
                    DbBulkAction.UPDATE,
                    _node_label(cls),
                    [normalize_model_row(cls, dict(row)) for row in rows],
                    _primary_key_fields(cls),
                    node_merge_policies(cls),
                )
                for cls, rows in rows_by_class.items()
            ]
        )
    def bulk_delete_nodes(self, model_class: Type[Any], pks: list[Any]) -> None:
        key_fields = _primary_key_fields(model_class)
        self._conn.bulk_write_nodes(
            DbBulkAction.DELETE,
            _node_label(model_class),
            [_pk_row(key_fields, pk) for pk in pks],
            key_fields,
            node_merge_policies(model_class),
        )
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
        self._new_ids().clear()
        self._dirty_ids().clear()
        self._deleted_ids().clear()
    def commit(self) -> None:
        self.flush()
        if self.expire_on_commit:
            self.expire_all()
    def rollback(self) -> None:
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()
        self._new_ids().clear()
        self._dirty_ids().clear()
        self._deleted_ids().clear()
        self._identity_map.clear()
        self._identity_keys_by_object_id.clear()
    def expire(self, instance: Any) -> None:
        if not getattr(self, "identity_tracking", True):
            return
        object_key = id(instance)
        identity_key = self._identity_keys_by_object_id.pop(object_key, None)
        if identity_key is not None:
            self._identity_map.pop(identity_key, None)
        current_key = self._identity_key_or_none(instance)
        if current_key is not None:
            self._identity_map.pop(current_key, None)
    def expire_all(self) -> None:
        self._identity_map.clear()
        self._identity_keys_by_object_id.clear()
    def close(self) -> None:
        self.rollback()
        if self._owns_connection:
            self._conn.close()
    def __enter__(self) -> "KuzuSession":
        return self
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False
    @contextmanager
    def begin(self) -> Iterator["KuzuSession"]:
        try:
            yield self
            self.commit()
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError, LookupError):
            self.rollback()
            raise
    def begin_nested(self) -> None:
        raise RuntimeError("Kuzu savepoint support is unavailable")
    def _flush_for_read(self) -> None:
        if self.autoflush and (self._new or self._dirty or self._deleted):
            self.flush()

    def _new_ids(self) -> set[int]:
        return self._pending_ids("_new_object_ids", self._new)

    def _dirty_ids(self) -> set[int]:
        return self._pending_ids("_dirty_object_ids", self._dirty)

    def _deleted_ids(self) -> set[int]:
        return self._pending_ids("_deleted_object_ids", self._deleted)

    def _pending_ids(self, attr: str, queue: list[Any]) -> set[int]:
        ids = getattr(self, attr, None)
        if ids is None:
            ids = {id(instance) for instance in queue}
            setattr(self, attr, ids)
        return ids

    @staticmethod
    def _append_pending(queue: list[Any], object_ids: set[int], instance: Any) -> None:
        object_id = id(instance)
        if object_id not in object_ids:
            queue.append(instance)
            object_ids.add(object_id)

    @staticmethod
    def _remove_pending(queue: list[Any], object_ids: set[int], instance: Any) -> None:
        object_id = id(instance)
        if object_id not in object_ids:
            return
        for index, pending in enumerate(queue):
            if pending is instance:
                del queue[index]
                object_ids.discard(object_id)
                return
        object_ids.discard(object_id)

    def _remember(self, instance: Any, *, track_identity: bool = True) -> None:
        if not track_identity or not getattr(self, "identity_tracking", True):
            return
        if hasattr(type(instance), "__kuzu_node_name__"):
            identity_key = self._identity_key_or_none(instance)
            if identity_key is None:
                return
            object_key = id(instance)
            old_key = self._identity_keys_by_object_id.get(object_key)
            if old_key is not None and old_key != identity_key:
                self._identity_map.pop(old_key, None)
            self._identity_map[identity_key] = instance
            self._identity_keys_by_object_id[object_key] = identity_key
    def _identity_key(self, instance: Any) -> str:
        key = self._identity_key_or_none(instance)
        if key is None:
            raise ValueError("identity key requires non-null primary key values")
        return key
    def _identity_key_or_none(self, instance: Any) -> str | None:
        cls = type(instance)
        key_fields = _primary_key_fields(cls)
        values = tuple(getattr(instance, field) for field in key_fields)
        if any(value is None for value in values):
            return None
        return f"{cls.__module__}.{cls.__qualname__}:{values!r}"
    def _write_instances(
        self,
        action: DbBulkAction,
        instances: list[Any],
        *,
        batch_size: int | None = None,
        track_identity: bool = True,
    ) -> None:
        if not instances:
            return
        self._write_instance_batch(action, instances, track_identity=track_identity)

    def _write_instance_batch(
        self,
        action: DbBulkAction,
        instances: list[Any],
        *,
        track_identity: bool,
    ) -> None:
        if action in {DbBulkAction.INSERT, DbBulkAction.CREATE}:
            self._validate_create_auto_increment_values(instances)
        if action in {DbBulkAction.INSERT, DbBulkAction.CREATE} and any(
            auto_generation_fields(instance) for instance in instances
        ):
            self._write_create_instances_with_generated_fields(
                instances,
                action=action,
                track_identity=track_identity,
            )
            return
        nodes, rels, identity_actions = self._partition_instance_rows(
            action,
            instances,
            track_identity=track_identity,
        )
        self._apply_identity_actions(identity_actions)
        self._write_grouped_rows(action, nodes, rels)

    def _validate_create_auto_increment_values(self, instances: list[Any]) -> None:
        for instance in instances:
            getter = getattr(instance, "get_manual_auto_increment_values", None)
            if not callable(getter):
                continue
            manual_values = getter()
            self._validate_manual_auto_increment_values(manual_values, type(instance))
            validate_explicit_null_primary_keys(instance, manual_values)

    def _validate_manual_auto_increment_values(
        self,
        manual_values: dict[str, Any],
        model_class: Type[Any],
    ) -> None:
        validate_kuzu_manual_auto_increment_values(
            manual_values,
            model_field_specs(model_class),
            model_class.__name__,
        )
    def _write_create_instances_with_generated_fields(
        self,
        instances: list[Any],
        *,
        action: DbBulkAction,
        track_identity: bool,
    ) -> None:
        normal_nodes: dict[type[Any], list[dict[str, Any]]] = {}
        generated_nodes: list[Any] = []
        normal_relationships: list[Any] = []
        generated_relationships: list[Any] = []
        for instance in instances:
            cls = type(instance)
            generated_fields = auto_generation_fields(instance)
            if hasattr(cls, "__kuzu_node_name__"):
                if generated_fields:
                    generated_nodes.append(instance)
                else:
                    normal_nodes.setdefault(cls, []).append(_node_row(instance))
                    self._remember(instance, track_identity=track_identity)
            elif hasattr(cls, "__kuzu_rel_name__"):
                if generated_fields:
                    generated_relationships.append(instance)
                else:
                    normal_relationships.append(instance)
            else:
                raise TypeError(f"{cls.__name__} is not a registered Kuzu model")
        self._write_grouped_rows(action, normal_nodes, {})
        self._write_generated_nodes(generated_nodes, track_identity=track_identity)
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        for instance in normal_relationships:
            row = _relationship_row(instance)
            rels.setdefault(_relationship_route(type(instance), row), []).append(row)
        self._write_grouped_rows(action, {}, rels)
        self._write_generated_relationships(generated_relationships)
    def _write_generated_nodes(self, instances: list[Any], *, track_identity: bool) -> None:
        if not instances:
            return
        payloads = self._conn.create_generated_nodes(
            [node_create_spec(instance) for instance in instances]
        )
        for instance, payload in zip(instances, payloads):
            apply_generated_values(instance, payload, auto_generation_fields(instance))
            self._remember(instance, track_identity=track_identity)
    def _write_generated_relationships(self, instances: list[Any]) -> None:
        if not instances:
            return
        payloads = self._conn.create_generated_relationships(
            [relationship_create_spec(instance) for instance in instances]
        )
        for instance, payload in zip(instances, payloads):
            apply_generated_values(instance, payload, auto_generation_fields(instance))
    def _write_instance_groups(
        self,
        action: DbBulkAction,
        node_instances: list[Any],
        relationship_instances: list[Any],
        *,
        track_identity: bool = True,
    ) -> None:
        for instance in node_instances:
            if not hasattr(type(instance), "__kuzu_node_name__"):
                raise TypeError(f"{type(instance).__name__} is not a registered Kuzu node")
        for instance in relationship_instances:
            if not hasattr(type(instance), "__kuzu_rel_name__"):
                raise TypeError(
                    f"{type(instance).__name__} is not a registered Kuzu relationship"
                )
        nodes, rels, identity_actions = self._partition_instance_rows(
            action,
            [*node_instances, *relationship_instances],
            track_identity=track_identity,
        )
        self._apply_identity_actions(identity_actions)
        self._write_grouped_rows(action, nodes, rels)

    def _partition_instance_rows(
        self,
        action: DbBulkAction,
        instances: list[Any],
        *,
        track_identity: bool,
    ) -> BulkRowPartition:
        worker_count = self._row_partition_worker_count(len(instances))
        if worker_count == 1:
            return self._partition_instance_row_range(
                action,
                instances,
                0,
                len(instances),
                track_identity=track_identity,
            )
        chunk_size = self._row_partition_chunk_size(len(instances), worker_count)
        nodes: dict[type[Any], list[dict[str, Any]]] = {}
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        identity_actions: list[BulkIdentityAction] = []
        partitions = self._conn.run_row_partition_tasks(
            [
                (
                    self._partition_instance_row_range,
                    (
                        action,
                        instances,
                        start,
                        min(len(instances), start + chunk_size),
                    ),
                    {"track_identity": track_identity},
                )
                for start in range(0, len(instances), chunk_size)
            ]
        )
        for chunk_nodes, chunk_rels, chunk_identity_actions in partitions:
            self._append_partition_rows(nodes, chunk_nodes)
            self._append_partition_rows(rels, chunk_rels)
            identity_actions.extend(chunk_identity_actions)
        return nodes, rels, identity_actions

    @staticmethod
    def _row_partition_chunk_size(total_instances: int, worker_count: int) -> int:
        if isinstance(total_instances, bool) or not isinstance(total_instances, int):
            raise TypeError("total_instances must be an integer")
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise TypeError("worker_count must be an integer")
        if total_instances < 1:
            raise ValueError("total_instances must be positive")
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        return (total_instances + worker_count - 1) // worker_count

    def _row_partition_worker_count(self, total_instances: int) -> int:
        if isinstance(total_instances, bool) or not isinstance(total_instances, int):
            raise TypeError("total_instances must be an integer")
        if total_instances <= 1:
            return 1
        if isinstance(self.bulk_batch_size, bool) or not isinstance(self.bulk_batch_size, int):
            raise TypeError("bulk_batch_size must be an integer")
        if self.bulk_batch_size < 1:
            raise ValueError("bulk_batch_size must be positive")
        cpu_count = os.process_cpu_count()
        if cpu_count is None or cpu_count < 1:
            raise RuntimeError("process CPU count is unavailable")
        batch_limited_workers = self._row_partition_chunk_size(
            total_instances,
            self.bulk_batch_size,
        )
        return min(cpu_count, total_instances, batch_limited_workers)

    @staticmethod
    def _partition_instance_row_range(
        action: DbBulkAction,
        instances: list[Any],
        start: int,
        stop: int,
        *,
        track_identity: bool,
    ) -> BulkRowPartition:
        nodes: dict[type[Any], list[dict[str, Any]]] = {}
        rels: dict[RelationshipRoute, list[dict[str, Any]]] = {}
        identity_actions: list[BulkIdentityAction] = []
        for index in range(start, stop):
            instance = instances[index]
            cls = type(instance)
            if hasattr(cls, "__kuzu_node_name__"):
                row = (
                    _node_delete_row(instance)
                    if action == DbBulkAction.DELETE
                    else _node_row(instance)
                )
                nodes.setdefault(cls, []).append(row)
                if track_identity:
                    if action == DbBulkAction.DELETE:
                        identity_actions.append(("expire", instance))
                    else:
                        identity_actions.append(("remember", instance))
            elif hasattr(cls, "__kuzu_rel_name__"):
                row = _relationship_row(instance)
                rels.setdefault(_relationship_route(cls, row), []).append(row)
            else:
                raise TypeError(f"{cls.__name__} is not a registered Kuzu model")
        return nodes, rels, identity_actions

    @staticmethod
    def _append_partition_rows(
        target: dict[Any, list[dict[str, Any]]],
        source: dict[Any, list[dict[str, Any]]],
    ) -> None:
        for key, rows in source.items():
            target.setdefault(key, []).extend(rows)

    def _apply_identity_actions(self, actions: list[BulkIdentityAction]) -> None:
        for action, instance in actions:
            if action == "expire":
                self.expire(instance)
            elif action == "remember":
                self._remember(instance)
            else:
                raise RuntimeError(f"unknown identity action: {action}")
    def _write_grouped_rows(
        self,
        action: DbBulkAction,
        nodes: dict[type[Any], list[dict[str, Any]]],
        rels: dict[RelationshipRoute, list[dict[str, Any]]],
    ) -> None:
        self._conn.bulk_write_nodes_and_relationships_many(
            [
                (action, _node_label(cls), rows, _primary_key_fields(cls), node_merge_policies(cls))
                for cls, rows in nodes.items()
            ],
            [
                (action, rel_type, from_label, to_label, rows, [from_key], [to_key])
                for (rel_type, from_label, to_label, from_key, to_key), rows in rels.items()
            ],
        )
