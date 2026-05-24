# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Type
from pydantic import model_validator

from .kuzu_orm import KuzuBaseModel
from .uuid_normalization import normalize_uuid_fields_for_model


class BaseModel(KuzuBaseModel):
    """Normalize ORM model inputs before Pydantic field validation."""

    @model_validator(mode='before')
    @classmethod
    def normalize_uuid_fields(cls: Type['BaseModel'], values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        return normalize_uuid_fields_for_model(model_class=cls, data=values)
