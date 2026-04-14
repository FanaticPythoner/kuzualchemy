from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Callable, Mapping, TypeVar, cast

from .constants import KuzuDataType

EnumType = TypeVar("EnumType", bound=Enum)

_INTEGER_STORAGE_TYPES = frozenset(
    {
        KuzuDataType.INT8,
        KuzuDataType.INT16,
        KuzuDataType.INT32,
        KuzuDataType.INT64,
        KuzuDataType.INT128,
        KuzuDataType.UINT8,
        KuzuDataType.UINT16,
        KuzuDataType.UINT32,
        KuzuDataType.UINT64,
        KuzuDataType.SERIAL,
    }
)


def _resolved_enum_factory(
    storage_type: KuzuDataType | None,
    base_enum: type[Enum] | None,
) -> type[Enum]:
    if storage_type is None:
        if base_enum is not None and issubclass(base_enum, IntEnum):
            return IntEnum
        return Enum
    if storage_type in _INTEGER_STORAGE_TYPES:
        return IntEnum
    return Enum


def _validate_base_enum_compatibility(
    storage_type: KuzuDataType | None,
    base_enum: type[Enum] | None,
) -> None:
    if base_enum is None:
        return
    enum_factory = _resolved_enum_factory(storage_type, base_enum)
    if enum_factory is IntEnum and not issubclass(base_enum, IntEnum):
        raise TypeError(f"{base_enum!r} is not an IntEnum")
    inherited_storage_type = getattr(base_enum, "__kuzu_enum_storage_type__", None)
    if inherited_storage_type is not None and storage_type is not None and inherited_storage_type != storage_type:
        raise ValueError(
            f"Kuzu enum storage type mismatch for {base_enum.__name__}: "
            f"expected {inherited_storage_type}, got {storage_type}"
        )


def _declared_class_members(enum_cls: type[Any]) -> dict[str, Any]:
    declared_members: dict[str, Any] = {}
    for name, value in vars(enum_cls).items():
        if name in {"__module__", "__qualname__", "__doc__", "__annotations__"} or name.startswith("_"):
            continue
        if callable(value) or isinstance(value, (staticmethod, classmethod, property)):
            raise TypeError(f"Kuzu enum decorator classes may only declare enum members; unsupported attribute: {name}")
        declared_members[name] = value
    return declared_members


def _stamp_kuzu_enum(
    enum_cls: type[EnumType],
    *,
    storage_type: KuzuDataType | None,
    base_enum: type[Enum] | None,
) -> type[EnumType]:
    setattr(enum_cls, "__kuzu_enum_extensible__", True)
    if storage_type is not None:
        setattr(enum_cls, "__kuzu_enum_storage_type__", storage_type)
    if base_enum is not None:
        setattr(enum_cls, "__kuzu_enum_base__", base_enum)
    return enum_cls


def _build_kuzu_enum_type(
    enum_name: str,
    base_enum: type[Enum] | None,
    members: Mapping[str, Any],
    *,
    storage_type: KuzuDataType | None,
    module: str,
    docstring: str | None,
) -> type[Enum]:
    _validate_base_enum_compatibility(storage_type, base_enum)
    base_members = {} if base_enum is None else {name: member.value for name, member in base_enum.__members__.items()}
    extra_members = dict(members)
    duplicate_names = set(base_members) & set(extra_members)
    if duplicate_names:
        raise ValueError(f"Duplicate enum member names for {enum_name}: {sorted(duplicate_names)}")
    if len(set(extra_members.values())) != len(extra_members):
        raise ValueError(f"Duplicate enum member values declared for {enum_name}")
    base_values = set() if base_enum is None else set(base_enum._value2member_map_)
    duplicate_values = [value for value in extra_members.values() if value in base_values]
    if duplicate_values:
        raise ValueError(f"Duplicate enum member values for {enum_name}: {duplicate_values}")
    enum_factory = _resolved_enum_factory(storage_type, base_enum)
    created_enum = enum_factory(enum_name, {**base_members, **extra_members}, module=module)
    created_enum.__doc__ = docstring or (None if base_enum is None else base_enum.__doc__)
    return _stamp_kuzu_enum(created_enum, storage_type=storage_type, base_enum=base_enum)


def _decorate_kuzu_enum(
    enum_cls: type[Any],
    *,
    storage_type: KuzuDataType,
    base_enum: type[Enum] | None,
) -> type[Enum]:
    if base_enum is None and issubclass(enum_cls, Enum):
        _validate_base_enum_compatibility(storage_type, enum_cls)
        return _stamp_kuzu_enum(enum_cls, storage_type=storage_type, base_enum=None)
    if base_enum is not None and issubclass(enum_cls, Enum):
        raise TypeError(f"{enum_cls.__name__} cannot both inherit Enum and extend {base_enum.__name__}")
    return _build_kuzu_enum_type(
        enum_cls.__name__,
        base_enum,
        _declared_class_members(enum_cls),
        storage_type=storage_type,
        module=enum_cls.__module__,
        docstring=enum_cls.__doc__,
    )


def kuzu_enum(
    storage_type: KuzuDataType,
    *,
    base_enum: type[Enum] | None = None,
) -> Callable[[type[Any]], type[Enum]]:
    def decorator(enum_cls: type[Any]) -> type[Enum]:
        return _decorate_kuzu_enum(enum_cls, storage_type=storage_type, base_enum=base_enum)

    return decorator


def kuzu_int_enum(
    storage_type: KuzuDataType,
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    if storage_type not in _INTEGER_STORAGE_TYPES:
        raise ValueError(f"{storage_type} is not an integer-backed Kuzu enum storage type")
    decorator = kuzu_enum(storage_type, base_enum=base_enum)
    if enum_cls is None:
        return decorator
    return decorator(enum_cls)


def kuzu_int8enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.INT8, enum_cls, base_enum=base_enum)


def kuzu_int16enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.INT16, enum_cls, base_enum=base_enum)


def kuzu_int32enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.INT32, enum_cls, base_enum=base_enum)


def kuzu_int64enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.INT64, enum_cls, base_enum=base_enum)


def kuzu_int128enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.INT128, enum_cls, base_enum=base_enum)


def kuzu_uint8enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.UINT8, enum_cls, base_enum=base_enum)


def kuzu_uint16enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.UINT16, enum_cls, base_enum=base_enum)


def kuzu_uint32enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.UINT32, enum_cls, base_enum=base_enum)


def kuzu_uint64enum(
    enum_cls: type[Any] | None = None,
    *,
    base_enum: type[IntEnum] | None = None,
) -> Callable[[type[Any]], type[Enum]] | type[Enum]:
    return kuzu_int_enum(KuzuDataType.UINT64, enum_cls, base_enum=base_enum)


def extend_enum(
    enum_name: str,
    base_enum: type[EnumType],
    members: Mapping[str, Any],
    *,
    module: str | None = None,
) -> type[Enum]:
    storage_type = getattr(base_enum, "__kuzu_enum_storage_type__", None)
    return cast(
        type[Enum],
        _build_kuzu_enum_type(
            enum_name,
            base_enum,
            members,
            storage_type=storage_type,
            module=module or base_enum.__module__,
            docstring=base_enum.__doc__,
        ),
    )


def extend_int_enum(
    enum_name: str,
    base_enum: type[IntEnum],
    members: Mapping[str, int],
    *,
    module: str | None = None,
) -> type[IntEnum]:
    if not issubclass(base_enum, IntEnum):
        raise TypeError(f"{base_enum!r} is not an IntEnum")
    return cast(type[IntEnum], extend_enum(enum_name, base_enum, members, module=module))


KUZU_INT8ENUM = kuzu_int8enum
KUZU_INT16ENUM = kuzu_int16enum
KUZU_INT32ENUM = kuzu_int32enum
KUZU_INT64ENUM = kuzu_int64enum
KUZU_INT128ENUM = kuzu_int128enum
KUZU_UINT8ENUM = kuzu_uint8enum
KUZU_UINT16ENUM = kuzu_uint16enum
KUZU_UINT32ENUM = kuzu_uint32enum
KUZU_UINT64ENUM = kuzu_uint64enum


__all__ = [
    "kuzu_enum",
    "kuzu_int_enum",
    "kuzu_int8enum",
    "kuzu_int16enum",
    "kuzu_int32enum",
    "kuzu_int64enum",
    "kuzu_int128enum",
    "kuzu_uint8enum",
    "kuzu_uint16enum",
    "kuzu_uint32enum",
    "kuzu_uint64enum",
    "KUZU_INT8ENUM",
    "KUZU_INT16ENUM",
    "KUZU_INT32ENUM",
    "KUZU_INT64ENUM",
    "KUZU_INT128ENUM",
    "KUZU_UINT8ENUM",
    "KUZU_UINT16ENUM",
    "KUZU_UINT32ENUM",
    "KUZU_UINT64ENUM",
    "extend_enum",
    "extend_int_enum",
]
