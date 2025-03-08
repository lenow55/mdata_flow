from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor


class ScopedABSUploaderVisitor(TypedDatasetVisitor, ABC):
    """
    Загружает превью датасета
    """

    _work_scope: dict[str, str] | None = None

    _run: Run | None = None
    _client: MlflowClient | None = None

    _root_artifact_path: str

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def set_scope(self, value: dict[str, str]):
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

    @contextmanager
    @abstractmethod
    def _manage_path(self, scope: str) -> Iterator[None]:
        pass

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

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for value in elem.datasets:
            with self._manage_path(value.name):
                # Если имени датасета нет в scope, то пропускаем его
                if self._work_scope and value.name not in self._work_scope:
                    continue
                value.Accept(visitor=self)
