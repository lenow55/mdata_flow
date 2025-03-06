import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import final

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.utils.dashboard import SaveMode
from mlflow.client import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor


class EvidentlyReportVisitor(TypedDatasetVisitor, ABC):
    """
    Рассчитывает отчёты evidently
    """

    # _results: dict[str, list[Report]] | None = None
    _work_scope: dict[str, str] | None = None
    _root_artifact_path: str = "reports"
    _tempdir = TemporaryDirectory(delete=False)

    def __init__(
        self,
        column_maping: ColumnMapping,
    ) -> None:
        super().__init__()
        self._column_maping: ColumnMapping = column_maping

    # def set_saver(self, value: dict[str, list[Report]]):
    #     self._results = value
    #
    def __del__(self):
        self._tempdir.cleanup()

    def set_scope(self, value: dict[str, str]):
        self._work_scope = value

    def set_client(self, client: MlflowClient):
        self._client = client

    @property
    def run(self):
        """The run property."""
        if not isinstance(self._run, Run):
            raise RuntimeError("Set run first")
        return self._run

    @run.setter
    def run(self, value: Run):
        self._run = value

    @abstractmethod
    def _pandas_build_report(self) -> Report | None: ...

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        # if not isinstance(self._results, dict):
        #     raise RuntimeError("Init results first")

        # if elem.name not in self._results:
        #     self._results.update({elem.name: []})

        report = self._pandas_build_report()
        if not report:
            return
        report.run(
            reference_data=None,
            current_data=elem.getDataset(),
            column_mapping=self._column_maping,
        )
        # self._results[elem.name].append(report)

        local_path = os.path.join(self._tempdir.name, f"{report.name}.html")
        report.save_html(
            filename=local_path,
            mode=SaveMode.SINGLE_FILE,
        )

        self._client.log_artifact(
            run_id=self.run.info.run_id,
            local_path=local_path,
            artifact_path=self._root_artifact_path,
        )

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        # if not isinstance(self._results, dict):
        #     raise RuntimeError("Init results first")

        for name, value in elem.datasets.items():
            self._root_artifact_path = os.path.join(self._root_artifact_path, name)
            if self._work_scope and name in self._work_scope:
                value.Accept(visitor=self)
            self._root_artifact_path = os.path.dirname(self._root_artifact_path)
