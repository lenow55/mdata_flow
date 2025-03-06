import os
from typing import final
from mlflow import MlflowClient
from mlflow.entities import Run
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor


class PreviewUploaderVisitor(TypedDatasetVisitor):
    """
    Загружает превью датасета
    """

    _work_scope: dict[str, str] | None = None

    _run: Run | None = None
    _client: MlflowClient | None = None

    _root_artifact_path: str = "previews"

    def __init__(
        self,
        count: int = 15,
    ) -> None:
        super().__init__()
        self._count_preview: int = count

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

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        if not isinstance(self._client, MlflowClient):
            raise RuntimeError("Setup client first")

        artifact_uri = self.run.info.artifact_uri

        if not isinstance(artifact_uri, str):
            raise RuntimeError(f"Bad artifact_uri {artifact_uri}")

        head_preview = elem.getDataset().head(self._count_preview)

        self._client.log_table(
            run_id=self.run.info.run_id,
            data=head_preview,
            artifact_file=os.path.join(self._root_artifact_path, f"{elem.name}.json"),
        )

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for name, value in elem.datasets.items():
            if self._work_scope and name in self._work_scope:
                value.Accept(visitor=self)
