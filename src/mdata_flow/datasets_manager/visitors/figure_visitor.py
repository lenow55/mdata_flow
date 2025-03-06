from abc import ABC, abstractmethod
from typing import final
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact


class FigureVisitor(TypedDatasetVisitor, ABC):
    """
    Базовый класс для визиторов генери
    """

    _results: dict[str, list[FigureArtifact]] | None = None
    _work_scope: dict[str, str] | None = None

    def __init__(
        self,
        plot_size: tuple[int, int] = (800, 600),
    ) -> None:
        super().__init__()
        self._plot_size: tuple[int, int] = plot_size

    def set_saver(self, value: dict[str, list[FigureArtifact]]):
        self._results = value

    def set_scope(self, value: dict[str, str]):
        self._work_scope = value

    @abstractmethod
    def _pandas_plot_figure(self, elem: PdDataset) -> FigureArtifact | None:
        pass

    @final
    @override
    def VisitPdDataset(self, elem: PdDataset):
        if not isinstance(self._results, dict):
            raise RuntimeError("Init results first")

        if elem.name not in self._results:
            self._results.update({elem.name: []})

        figure = self._pandas_plot_figure(elem)

        if figure:
            self._results[elem.name].append(figure)

    @final
    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        if not isinstance(self._results, dict):
            raise RuntimeError("Init results first")

        for name, value in elem.datasets.items():
            if self._work_scope and name in self._work_scope:
                value.Accept(visitor=self)
