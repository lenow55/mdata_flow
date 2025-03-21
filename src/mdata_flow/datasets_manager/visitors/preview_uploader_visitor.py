import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

from mlflow import MlflowClient
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.visitors.scoped_abs_info_uploader import (
    ScopedABSUploaderVisitor,
)


@final
class PreviewUploaderVisitor(ScopedABSUploaderVisitor):
    """
    Загружает превью датасета
    """

    _root_artifact_path: str = "previews"
    _count_preview: int

    def __init__(
        self,
        count: int = 15,
    ) -> None:
        super().__init__()
        self._count_preview = count

    @final
    @contextmanager
    def _manage_path(self, elem: IDataset) -> Iterator[None]:
        try:
            if isinstance(elem, GroupDataset):
                self._root_artifact_path = os.path.join(
                    self._root_artifact_path, elem.name
                )
            self._current_ds_key_path.append(elem.name)
            yield
        finally:
            if isinstance(elem, GroupDataset):
                self._root_artifact_path = os.path.dirname(self._root_artifact_path)
            _ = self._current_ds_key_path.pop()

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
