import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.scoped_abs_info_uploader import (
    ScopedABSUploaderVisitor,
)
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact


class FigureVisitor(ScopedABSUploaderVisitor, ABC):
    """
    Базовый класс для визиторов генерации графиков
    по датасету
    Загружает графики сразу
    """

    def __init__(
        self,
        plot_size: tuple[int, int] = (800, 600),
    ) -> None:
        super().__init__()
        self._plot_size: tuple[int, int] = plot_size

    @final
    @contextmanager
    def _manage_path(self, scope: str) -> Iterator[None]:
        try:
            self._root_artifact_path: str = os.path.join(
                self._root_artifact_path, scope
            )
            yield
        finally:
            self._root_artifact_path = os.path.dirname(self._root_artifact_path)

    @abstractmethod
    def _pandas_plot_figure(self, elem: PdDataset) -> FigureArtifact | None:
        pass

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        figure = self._pandas_plot_figure(elem)

        if figure:
            self.client.log_figure(
                run_id=self.run.info.run_id,
                figure=figure["plot"],
                artifact_file=os.path.join(
                    self._root_artifact_path, figure["artifact_name"]
                ),
            )
