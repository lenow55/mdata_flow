from typing import TypeAlias, TypeVar, Union

T = TypeVar("T")
NestedDict: TypeAlias = dict[str, Union[T, "NestedDict[T]"]]
