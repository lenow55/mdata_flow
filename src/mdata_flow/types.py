from typing import TypeAlias, TypeVar

T = TypeVar("T")
NestedDict: TypeAlias = dict[str, "T| NestedDict[T]"]
