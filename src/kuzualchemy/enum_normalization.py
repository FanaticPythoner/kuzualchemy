from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import types
from typing import Any, Dict, Type, Union, get_args, get_origin
from weakref import WeakKeyDictionary

_ENUM_CACHE: Dict[Type[Enum], tuple[Dict[str, Enum], Dict[Any, Enum]]] = {}
_MODEL_ENUM_PLAN_CACHE: "WeakKeyDictionary[type[Any], tuple[EnumFieldConversionPlan, ...]]" = WeakKeyDictionary()
_MISSING = object()
_NONE_TYPE = type(None)
_UNION_ORIGINS = (Union, types.UnionType) if hasattr(types, "UnionType") else (Union,)


@dataclass(frozen=True)
class EnumConversionBranch:
    is_sequence: bool
    enum_types: tuple[Type[Enum], ...]
    allow_none_elements: bool = False
    allow_passthrough_elements: bool = False


@dataclass(frozen=True)
class EnumFieldConversionPlan:
    field_name: str
    branches: tuple[EnumConversionBranch, ...]
    all_enum_types: tuple[Type[Enum], ...]
    allow_none_value: bool
    allow_passthrough: bool


def get_enum_lookups(enum_type: Type[Enum]) -> tuple[Dict[str, Enum], Dict[Any, Enum]]:
    if enum_type not in _ENUM_CACHE:
        names: Dict[str, Enum] = {}
        values: Dict[Any, Enum] = {}
        for member in enum_type:
            names[member.name] = member
            values[member.value] = member
        _ENUM_CACHE[enum_type] = (names, values)
    return _ENUM_CACHE[enum_type]


def create_enum_converter(enum_type: Type[Enum]):
    def convert_element(elem: Any) -> Any:
        member = _lookup_enum_value(enum_type, elem)
        if member is not _MISSING:
            return member
        raise _build_enum_error(enum_type.__name__, (enum_type,), elem)

    return convert_element


def clear_model_enum_conversion_plan(model_class: type[Any]) -> None:
    _MODEL_ENUM_PLAN_CACHE.pop(model_class, None)


def clear_all_enum_conversion_plans() -> None:
    _MODEL_ENUM_PLAN_CACHE.clear()


def _try_value_lookup(value_map: Dict[Any, Enum], raw_value: Any) -> Enum | object:
    try:
        return value_map.get(raw_value, _MISSING)
    except TypeError:
        return _MISSING


def _lookup_enum_value(enum_type: Type[Enum], raw_value: Any) -> Enum | object:
    names, value_map = get_enum_lookups(enum_type)
    if isinstance(raw_value, Enum):
        if raw_value.__class__ is enum_type:
            return raw_value
        member = _try_value_lookup(value_map, raw_value.value)
        if member is not _MISSING:
            return member
        member = names.get(raw_value.name, _MISSING)
        if member is not _MISSING:
            return member
    member = _try_value_lookup(value_map, raw_value)
    if member is not _MISSING:
        return member
    if isinstance(raw_value, str):
        member = names.get(raw_value, _MISSING)
        if member is not _MISSING:
            return member
        if len(raw_value) > 0:
            first_char = raw_value[0]
            if first_char.isdigit() or (len(raw_value) > 1 and first_char == '-' and raw_value[1].isdigit()):
                try:
                    numeric = int(raw_value) if '.' not in raw_value and 'e' not in raw_value.lower() else float(raw_value)
                    member = _try_value_lookup(value_map, numeric)
                    if member is not _MISSING:
                        return member
                except (ValueError, OverflowError):
                    pass
    return _MISSING


def _build_enum_error(field_name: str, enum_types: tuple[Type[Enum], ...], raw_value: Any) -> ValueError:
    valid_name_set: set[str] = set()
    valid_names: list[str] = []
    valid_values: list[Any] = []
    for enum_type in enum_types:
        names, value_map = get_enum_lookups(enum_type)
        for name in names.keys():
            if name not in valid_name_set:
                valid_name_set.add(name)
                valid_names.append(name)
        for value in value_map.keys():
            if value is None:
                continue
            if value not in valid_values:
                valid_values.append(value)
    return ValueError(
        f"Invalid value for field {field_name}: {raw_value} "
        f"Valid names: {valid_names}, valid values: {valid_values}"
    )


def _extract_direct_enum_types(args: tuple[Any, ...]) -> tuple[Type[Enum], ...]:
    enum_types: list[Type[Enum]] = []
    for candidate in args:
        if isinstance(candidate, type) and issubclass(candidate, Enum):
            enum_types.append(candidate)
    return tuple(enum_types)


def _try_convert_multi_enum_value(enum_types: tuple[Type[Enum], ...], raw_value: Any) -> Enum | object:
    for enum_type in enum_types:
        member = _lookup_enum_value(enum_type, raw_value)
        if member is not _MISSING:
            return member
    return _MISSING


def _is_union_origin(origin: Any) -> bool:
    return origin in _UNION_ORIGINS


def _build_sequence_element_branch(annotation: Any) -> EnumConversionBranch | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return EnumConversionBranch(is_sequence=True, enum_types=(annotation,))
    origin = get_origin(annotation)
    if not _is_union_origin(origin):
        return None
    union_args = get_args(annotation)
    enum_types = _extract_direct_enum_types(union_args)
    if not enum_types:
        return None
    non_none_args = tuple(candidate for candidate in union_args if candidate is not _NONE_TYPE)
    return EnumConversionBranch(
        is_sequence=True,
        enum_types=enum_types,
        allow_none_elements=_NONE_TYPE in union_args,
        allow_passthrough_elements=len(enum_types) != len(non_none_args),
    )


def _build_tuple_sequence_branch(tuple_args: tuple[Any, ...]) -> EnumConversionBranch | None:
    if not tuple_args:
        return None
    if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
        return _build_sequence_element_branch(tuple_args[0])
    first_branch = _build_sequence_element_branch(tuple_args[0])
    if first_branch is None:
        return None
    for candidate in tuple_args[1:]:
        if _build_sequence_element_branch(candidate) != first_branch:
            return None
    return first_branch


def _build_non_union_branch(annotation: Any) -> EnumConversionBranch | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return EnumConversionBranch(is_sequence=False, enum_types=(annotation,))
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        if not args:
            return None
        return _build_sequence_element_branch(args[0])
    if origin is tuple:
        return _build_tuple_sequence_branch(get_args(annotation))
    return None


def _collect_plan_enum_types(branches: tuple[EnumConversionBranch, ...]) -> tuple[Type[Enum], ...]:
    ordered: list[Type[Enum]] = []
    for branch in branches:
        for enum_type in branch.enum_types:
            if enum_type not in ordered:
                ordered.append(enum_type)
    return tuple(ordered)


def _build_field_conversion_plan(field_name: str, annotation: Any) -> EnumFieldConversionPlan | None:
    origin = get_origin(annotation)
    if not _is_union_origin(origin):
        branch = _build_non_union_branch(annotation)
        if branch is None:
            return None
        branches = (branch,)
        return EnumFieldConversionPlan(
            field_name=field_name,
            branches=branches,
            all_enum_types=_collect_plan_enum_types(branches),
            allow_none_value=False,
            allow_passthrough=False,
        )
    union_args = get_args(annotation)
    branches_list: list[EnumConversionBranch] = []
    allow_passthrough = False
    for candidate in union_args:
        if candidate is _NONE_TYPE:
            continue
        branch = _build_non_union_branch(candidate)
        if branch is None:
            allow_passthrough = True
            continue
        if branch not in branches_list:
            branches_list.append(branch)
    if not branches_list:
        return None
    branches = tuple(branches_list)
    return EnumFieldConversionPlan(
        field_name=field_name,
        branches=branches,
        all_enum_types=_collect_plan_enum_types(branches),
        allow_none_value=_NONE_TYPE in union_args,
        allow_passthrough=allow_passthrough,
    )


def _build_model_enum_conversion_plans(model_class: type[Any]) -> tuple[EnumFieldConversionPlan, ...]:
    plans: list[EnumFieldConversionPlan] = []
    for field_name, field_info in model_class.model_fields.items():
        plan = _build_field_conversion_plan(field_name, field_info.annotation)
        if plan is not None:
            plans.append(plan)
    return tuple(plans)


def _get_model_enum_conversion_plans(model_class: type[Any]) -> tuple[EnumFieldConversionPlan, ...]:
    cached = _MODEL_ENUM_PLAN_CACHE.get(model_class)
    if cached is not None:
        return cached
    plans = _build_model_enum_conversion_plans(model_class)
    _MODEL_ENUM_PLAN_CACHE[model_class] = plans
    return plans


def _convert_sequence_branch(field_name: str, branch: EnumConversionBranch, raw_value: Any) -> list[Any] | tuple[Any, ...] | object:
    if not isinstance(raw_value, (list, tuple)):
        return _MISSING
    converted: list[Any] = []
    for element in raw_value:
        if branch.allow_none_elements and element is None:
            converted.append(None)
            continue
        member = _try_convert_multi_enum_value(branch.enum_types, element)
        if member is not _MISSING:
            converted.append(member)
            continue
        if branch.allow_passthrough_elements:
            converted.append(element)
            continue
        raise _build_enum_error(field_name, branch.enum_types, element)
    if isinstance(raw_value, tuple):
        return tuple(converted)
    return converted


def convert_input_enums_for_model(*, model_class: type[Any], values: Any) -> Any:
    if not isinstance(values, dict):
        return values
    plans = _get_model_enum_conversion_plans(model_class)
    if not plans:
        return values
    for plan in plans:
        value = values.get(plan.field_name, _MISSING)
        if value is _MISSING:
            continue
        if value is None and plan.allow_none_value:
            continue
        branches = plan.branches
        if len(branches) == 1:
            branch = branches[0]
            if branch.is_sequence:
                converted = _convert_sequence_branch(plan.field_name, branch, value)
                if converted is _MISSING:
                    if plan.allow_passthrough:
                        continue
                    raise _build_enum_error(plan.field_name, plan.all_enum_types, value)
                values[plan.field_name] = converted
                continue
            converted = _try_convert_multi_enum_value(branch.enum_types, value)
            if converted is not _MISSING:
                values[plan.field_name] = converted
                continue
            if plan.allow_passthrough:
                continue
            raise _build_enum_error(plan.field_name, plan.all_enum_types, value)
        converted_value: Any = _MISSING
        for branch in branches:
            if branch.is_sequence:
                converted_value = _convert_sequence_branch(plan.field_name, branch, value)
            else:
                converted_value = _try_convert_multi_enum_value(branch.enum_types, value)
            if converted_value is not _MISSING:
                values[plan.field_name] = converted_value
                break
        else:
            if plan.allow_passthrough:
                continue
            raise _build_enum_error(plan.field_name, plan.all_enum_types, value)
    return values
