import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import final

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.utils.dashboard import SaveMode
from mlflow.client import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.visitors.scoped_abs_info_uploader import (
    ScopedABSUploaderVisitor,
)


class EvidentlyReportVisitor(ScopedABSUploaderVisitor, ABC):
    """
    Рассчитывает отчёты evidently
    """

    _root_artifact_path: str = "reports"
    _tempdir = TemporaryDirectory(delete=False)

    def __init__(
        self,
        column_maping: ColumnMapping,
    ) -> None:
        super().__init__()
        self._column_maping: ColumnMapping = column_maping

    def __del__(self):
        self._tempdir.cleanup()

    @final
    @contextmanager
    def _manage_path(self, elem: IDataset) -> Iterator[None]:
        try:
            self._root_artifact_path = os.path.join(self._root_artifact_path, elem.name)
            self._current_ds_key_path.append(elem.name)
            yield
        finally:
            self._root_artifact_path = os.path.dirname(self._root_artifact_path)
            _ = self._current_ds_key_path.pop()

    @abstractmethod
    def _pandas_build_report(self) -> Report | None: ...

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        report = self._pandas_build_report()
        if not report:
            return
        report.run(  # pyright: ignore[reportArgumentType]
            reference_data=None,
            current_data=elem.getDataset(),
            column_mapping=self._column_maping,
        )

        local_path = os.path.join(self._tempdir.name, f"{report.name}.html")
        report.save_html(
            filename=local_path,
            mode=SaveMode.SINGLE_FILE,
        )

        self.client.log_artifact(
            run_id=self.run.info.run_id,
            local_path=local_path,
            artifact_path=self._root_artifact_path,
        )
