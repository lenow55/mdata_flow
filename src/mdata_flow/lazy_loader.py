import importlib
import types

from typing_extensions import Any, override


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
