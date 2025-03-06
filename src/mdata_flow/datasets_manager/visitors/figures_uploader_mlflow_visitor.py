import os

from mlflow.client import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact


class FiguresUploaderDatasetVisitor(TypedDatasetVisitor):
    """
    Загружает графики в mlflow
    """

    _run: Run | None = None

    _client: MlflowClient

    _root_artifact_path: str = "plots"
    _plots_store: dict[str, list[FigureArtifact]]

    def __init__(
        self,
        client: MlflowClient,
        plots_store: dict[str, list[FigureArtifact]],
        experiment_id: str,
    ) -> None:
        self._client = client
        self._plots_store = plots_store

    @property
    def run(self):
        """The run property."""
        if not isinstance(self._run, Run):
            raise RuntimeError("Set run first")
        return self._run

    @run.setter
    def run(self, value: Run):
        self._run = value

    @override
    def VisitPdDataset(self, elem: PdDataset):
        run = self.run

        if elem.name not in self._plots_store:
            return

        for figure in self._plots_store[elem.name]:
            self._client.log_figure(
                run_id=run.info.run_id,
                figure=figure["plot"],
                artifact_file=os.path.join(
                    self._root_artifact_path, figure["artifact_name"]
                ),
            )

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for _, dataset in elem.datasets.items():
            self._root_artifact_path = os.path.join(
                self._root_artifact_path, dataset.name
            )
            dataset.Accept(visitor=self)
            self._root_artifact_path = os.path.dirname(self._root_artifact_path)
