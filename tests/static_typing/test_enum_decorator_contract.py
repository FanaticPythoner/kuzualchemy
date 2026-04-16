from enum import IntEnum, StrEnum
from typing import Literal, assert_type

from kuzualchemy import KuzuDataType, kuzu_enum, kuzu_enum_member, kuzu_int8enum


@kuzu_enum(KuzuDataType.STRING)
class DirectText(StrEnum):
    ALPHA = "alpha"


class CanonicalText(StrEnum):
    BASE = "base"


class _ExtendedTextBase:
    @property
    def name(self) -> str: ...

    @property
    def value(self) -> str: ...

    BASE: "ExtendedText"


@kuzu_enum(KuzuDataType.STRING, base_enum=CanonicalText)
class ExtendedText(_ExtendedTextBase, StrEnum):
    CHILD = kuzu_enum_member("child")


class CanonicalSmallInt(IntEnum):
    BASE = 1


class _ExtendedSmallIntBase:
    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...

    BASE: "ExtendedSmallInt"


@kuzu_int8enum(base_enum=CanonicalSmallInt)
class ExtendedSmallInt(_ExtendedSmallIntBase, IntEnum):
    CHILD = kuzu_enum_member(2)


def _take_direct_text(value: DirectText) -> DirectText:
    return value


def _take_text(value: ExtendedText) -> ExtendedText:
    return value


def _take_small_int(value: ExtendedSmallInt) -> ExtendedSmallInt:
    return value


assert_type(_take_direct_text(DirectText.ALPHA), DirectText)
assert_type(ExtendedText.BASE, ExtendedText)
assert_type(ExtendedText.CHILD, Literal[ExtendedText.CHILD])
assert_type(ExtendedText.BASE.value, str)
assert_type(_take_text(ExtendedText.CHILD), ExtendedText)
assert_type(ExtendedSmallInt.BASE, ExtendedSmallInt)
assert_type(ExtendedSmallInt.CHILD, Literal[ExtendedSmallInt.CHILD])
assert_type(ExtendedSmallInt.BASE.value, int)
assert_type(_take_small_int(ExtendedSmallInt.CHILD), ExtendedSmallInt)
