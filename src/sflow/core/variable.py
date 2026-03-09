# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any

from pydantic import BaseModel


class VariableType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"

    def __str__(self) -> str:
        return self.value


class Variable(BaseModel):
    name: str
    value: Any
    description: str | None = None
    type: VariableType = VariableType.STRING
    domain: list[Any] | None = None
