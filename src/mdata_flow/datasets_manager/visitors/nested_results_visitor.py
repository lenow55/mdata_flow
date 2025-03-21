from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Generic, TypeVar, final

from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.types import NestedDict

T = TypeVar("T")


class NestedResultsDatasetVisitor(TypedDatasetVisitor, ABC, Generic[T]):
    _results: NestedDict[T]
    # ссылка на текущий корень обработки
    _results_tmp_link: NestedDict[T]
    # список ключей текущего уровня
    _current_ds_key_path: list[str]

    def __init__(self) -> None:
        super().__init__()
        self._results = {}
        self._results_tmp_link = self._results
        self._current_ds_key_path = []

    def get_results(self):
        return self._results

    @abstractmethod
    def _visit_pd_dataset(self, elem: PdDataset) -> T:
        pass

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        result = self._visit_pd_dataset(elem)
        # забираем текущий ключ из списка и по нему назначаем
        # результат
        try:
            key = self._current_ds_key_path[-1]
            self._results_tmp_link.update({key: result})
        except IndexError:
            # INFO: Посетитель обрабатывает только один датасет
            # просто добавим ключ в результат
            self._results_tmp_link.update({elem.name: result})

    @contextmanager
    def _manage_path(self) -> Iterator[None]:
        backup_tmp_link = self._results_tmp_link
        if len(self._current_ds_key_path):
            # если путь не пустой, значит вызваны из группы
            self._results_tmp_link.update({self._current_ds_key_path[-1]: {}})
            tmp_link = self._results_tmp_link[self._current_ds_key_path[-1]]
            if not isinstance(tmp_link, dict):
                raise RuntimeError(f"Bad tmp_link in Visitor {self.__class__.__name__}")

            # переносим ссылку на новую вложенность
            self._results_tmp_link = tmp_link

        yield

        if len(self._current_ds_key_path):
            self._results_tmp_link = backup_tmp_link

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        with self._manage_path():
            for value in elem.datasets:
                # добавляем ключ датасета, в который заходить будем
                self._current_ds_key_path.append(value.name)
                value.Accept(visitor=self)
                # извлекаем ключ, не нужен
                _ = self._current_ds_key_path.pop()
