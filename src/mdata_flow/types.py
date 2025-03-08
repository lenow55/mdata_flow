from typing import TypeAlias, Union

NestedDict: TypeAlias = dict[str, Union[str, "NestedDict"]]
