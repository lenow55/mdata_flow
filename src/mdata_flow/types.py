from typing import TypeAlias, TypeVar, Union

TResult = TypeVar("TResult")
TParam = TypeVar("TParam")

T = TypeVar("T")
NestedDict: TypeAlias = dict[str, Union[T, "NestedDict[T]"]]
