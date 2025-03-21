from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from functools import reduce
from typing import final

from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import Dataset, GroupDataset
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.types import NestedDict


class ScopedABSUploaderVisitor(TypedDatasetVisitor, ABC):
    """
    Абстрактный посетитель с возможностью указания scope.
    Не сохраняет результаты в самом себе
    """

    _work_scope: NestedDict[str | None] | None = None
    _current_ds_key_path: list[str] = []

    _run: Run | None = None
    _client: MlflowClient | None = None

    _root_artifact_path: str

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def set_scope(self, value: NestedDict[str | None]):
        """
        Устанавливает scope для выборки датасетов, к которым надо
        обработать доп инфу
        """
        self._work_scope = value

    @property
    def client(self) -> MlflowClient:
        """The client property."""
        if not isinstance(self._client, MlflowClient):
            raise ValueError("Set mlflow client first")
        return self._client

    @client.setter
    def client(self, value: MlflowClient):
        """
        Устанавливает клиента mlflow для данного загрузчика
        """
        self._client = value

    @property
    def run(self):
        """The run property."""
        if not isinstance(self._run, Run):
            raise RuntimeError("Set run first")
        return self._run

    @run.setter
    def run(self, value: Run):
        """
        Устанавливаем текущий ран, куда надо загрузить данные
        """
        self._run = value

    @contextmanager
    @abstractmethod
    def _manage_path(self, elem: IDataset) -> Iterator[None]:
        pass

    def _check_need_process(self, elem: IDataset) -> bool:
        if not self._work_scope:
            return True

        tmp_scope_link = self._work_scope
        for key in self._current_ds_key_path:
            value = tmp_scope_link[key]
            if isinstance(value, str) and isinstance(elem, Dataset):
                return True
            if isinstance(value, dict):
                if isinstance(elem, GroupDataset):
                    return True
                tmp_scope_link = value
                continue
            return False

        return False

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for value in elem.datasets:
            with self._manage_path(value):
                # Если имени датасета нет в scope, то пропускаем его
                if not self._check_need_process(value):
                    continue
                value.Accept(visitor=self)
