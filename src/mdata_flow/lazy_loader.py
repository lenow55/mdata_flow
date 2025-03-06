import importlib
import types

from typing import Any, override


class LazyLoader:
    """Класс-обёртка для ленивой загрузки модулей.
    Импортирует модуль при первом обращении к его атрибутам.
    """

    def __init__(self, modname: str) -> None:
        self._modname: str = modname
        self._mod: types.ModuleType | None = None

    def _load_module(self) -> types.ModuleType:
        """Импортирует модуль, если он ещё не загружен."""
        if self._mod is None:
            self._mod = importlib.import_module(self._modname)
        return self._mod

    def __getattr__(self, attr: str) -> Any:
        """Вызывается при обращении к атрибутам модуля."""
        module = self._load_module()
        return getattr(module, attr)

    @override
    def __dir__(self) -> list[str]:
        """Позволяет корректно работать с функциями типа dir()."""
        module = self._load_module()
        return dir(module)


# class LazyLoader:
#     "thin shell class to wrap modules.  load real module on first access and pass thru"
#
#     def __init__(self, modname: str):
#         self._modname: str = modname
#         self._mod: Any | None = None
#
#     def __getattr__(self, attr: Any):
#         "import module on first attribute access"
#
#         try:
#             return getattr(self._mod, attr)
#
#         except Exception as e:
#             if self._mod is None:
#                 self._mod = importlib.import_module(self._modname)
#             else:
#                 raise e
