from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, final

from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.types import NestedDict, TParam, TResult


class NestedDatasetVisitor(TypedDatasetVisitor, ABC, Generic[TParam, TResult]):
    # входные параметры
    _params: NestedDict[TParam]
    # ссылка на текущий корень обработки
    _params_tmp_link: NestedDict[TParam]

    _results: NestedDict[TResult]
    # ссылка на текущий корень обработки
    _results_tmp_link: NestedDict[TResult]
    # список ключей текущего уровня
    _current_ds_key_path: list[str]

    def __init__(self) -> None:
        super().__init__()
        self._results = {}
        self._results_tmp_link = self._results
        self._current_ds_key_path = []
        self._params = {}
        self._params_tmp_link = self._params

    def set_params(self, params: NestedDict[TParam]):
        self._params = params
        self._params_tmp_link = self._params

    def get_results(self):
        return self._results

    @abstractmethod
    def _visit_pd_dataset(self, elem: PdDataset) -> TResult:
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
        backup_tmp_params_link = self._params_tmp_link
        if len(self._current_ds_key_path):
            # если путь не пустой, значит вызваны из верхнеуровневой группы
            self._results_tmp_link.update({self._current_ds_key_path[-1]: {}})
            tmp_link = self._results_tmp_link[self._current_ds_key_path[-1]]
            if not isinstance(tmp_link, dict):
                raise RuntimeError(f"Bad tmp_link in Visitor {self.__class__.__name__}")

            # переносим ссылку на новую вложенность
            self._results_tmp_link = tmp_link

            try:
                tmp_params_link = self._params_tmp_link[self._current_ds_key_path[-1]]
            except KeyError:
                # raise RuntimeError(f"Bad params for this composite {self._params}")
                # INFO: установим пустой словарь, чтобы решение обрабатывать или нет принималось
                # на уровне датасета в реализации
                tmp_params_link = NestedDict[TParam]()
            if not isinstance(tmp_params_link, dict):
                raise RuntimeError(
                    f"Bad tmp_params_link in Visitor {self.__class__.__name__}"
                )
            self._params_tmp_link = tmp_params_link

        yield

        if len(self._current_ds_key_path):
            self._results_tmp_link = backup_tmp_link
            self._params_tmp_link = backup_tmp_params_link

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
