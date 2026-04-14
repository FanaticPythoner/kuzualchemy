from __future__ import annotations

from enum import IntEnum

from kuzualchemy import (
    BaseModel,
    KuzuDataType,
    KuzuRelationshipBase,
    get_ddl_for_node,
    get_ddl_for_relationship,
    kuzu_field,
    kuzu_field_add,
    kuzu_field_edit,
    kuzu_field_override,
    kuzu_field_remove,
    kuzu_int8enum,
    kuzu_node,
    kuzu_relationship,
)


@kuzu_int8enum
class BaseStatus(IntEnum):
    BASE = 0


@kuzu_int8enum(base_enum=BaseStatus)
class ExtendedStatus:
    CHILD = 1


@kuzu_node("DirectiveNodeBase", abstract=True)
class DirectiveNodeBase(BaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    legacy: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    status: BaseStatus = kuzu_field(kuzu_type=KuzuDataType.INT8, not_null=True)


@kuzu_node(
    "DirectiveNode",
    field_directives=[
        kuzu_field_add("extra_flag", bool, kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)),
        kuzu_field_override("summary", str, kuzu_field(kuzu_type=KuzuDataType.STRING, default="overridden")),
    ],
)
class DirectiveNode(DirectiveNodeBase):
    __kuzu_field_directives__ = (
        kuzu_field_remove("legacy"),
        kuzu_field_edit("status", annotation=ExtendedStatus, description="extended-status"),
    )
    summary: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="base")


@kuzu_relationship(abstract=True)
class DirectiveRelationshipBase(KuzuRelationshipBase):
    kind: BaseStatus = kuzu_field(kuzu_type=KuzuDataType.INT8, not_null=True)
    label: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)


@kuzu_relationship(
    "DirectiveRelationship",
    pairs=[(DirectiveNode, DirectiveNode)],
    field_directives=[
        kuzu_field_edit("kind", annotation=ExtendedStatus),
        kuzu_field_add("confidence", float, kuzu_field(kuzu_type=KuzuDataType.FLOAT, default=0.0)),
    ],
)
class DirectiveRelationship(DirectiveRelationshipBase):
    __kuzu_field_directives__ = (kuzu_field_remove("label"),)


def test_node_field_directives_update_effective_model_surface() -> None:
    node = DirectiveNode(id=1, status=BaseStatus.BASE)

    assert set(DirectiveNode.model_fields) == {"id", "status", "summary", "extra_flag"}
    assert DirectiveNode.model_fields["status"].annotation is ExtendedStatus
    assert DirectiveNode.model_fields["summary"].default == "overridden"
    assert node.status is getattr(ExtendedStatus, "BASE")
    assert node.summary == "overridden"
    assert node.extra_flag is False
    assert not hasattr(node, "legacy")


def test_relationship_field_directives_update_effective_model_surface() -> None:
    rel = DirectiveRelationship(from_node=1, to_node=2, kind=BaseStatus.BASE)

    assert set(DirectiveRelationship.model_fields) == {"from_node", "to_node", "kind", "confidence"}
    assert DirectiveRelationship.model_fields["kind"].annotation is ExtendedStatus
    assert rel.kind is getattr(ExtendedStatus, "BASE")
    assert rel.confidence == 0.0


def test_directive_ddl_generation_uses_effective_inherited_fields() -> None:
    node_ddl = get_ddl_for_node(DirectiveNode)
    relationship_ddl = get_ddl_for_relationship(DirectiveRelationship)

    assert "legacy" not in node_ddl
    assert "summary" in node_ddl
    assert "extra_flag" in node_ddl
    assert "label" not in relationship_ddl
    assert "kind INT8" in relationship_ddl
    assert "confidence FLOAT" in relationship_ddl


def test_extended_enum_members_convert_on_inherited_field_edits() -> None:
    node = DirectiveNode(id=2, status="CHILD")
    rel = DirectiveRelationship(from_node=1, to_node=2, kind="CHILD")

    assert node.status is ExtendedStatus.CHILD
    assert rel.kind is ExtendedStatus.CHILD


def test_kuzu_int8enum_decorator_marks_storage_and_base_contract() -> None:
    assert getattr(BaseStatus, "__kuzu_enum_storage_type__") == KuzuDataType.INT8
    assert getattr(ExtendedStatus, "__kuzu_enum_storage_type__") == KuzuDataType.INT8
    assert getattr(ExtendedStatus, "__kuzu_enum_base__") is BaseStatus
    assert getattr(ExtendedStatus, "BASE").value == BaseStatus.BASE.value
