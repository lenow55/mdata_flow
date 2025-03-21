from typing import final
import pandas as pd

from mlflow.types.schema import DataType
from typing_extensions import Any, override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.figure_visitor import FigureVisitor
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact
from mdata_flow.plotly_ext.func_library import plot_correlation_matrix


@final
class PlotlyCorrVisitor(FigureVisitor):
    """
    Рассчитывает матрицу корелляций для датасетов
    """

    def __init__(
        self,
        plot_size: tuple[int, int] = (800, 800),
    ) -> None:
        super().__init__(plot_size=plot_size)

    def _check_type(self, data_type: Any) -> bool:
        if not isinstance(data_type, DataType):
            return False

        return data_type in [
            DataType.boolean,
            DataType.integer,
            DataType.double,
            DataType.long,
            DataType.float,
        ]

    @final
    @override
    def _pandas_plot_figure(self, elem: PdDataset) -> FigureArtifact | None:
        dataset = elem.getDataset()
        available_cols = [
            col_name
            for col_name, col_type in elem.schema.input_types_dict().items()
            if self._check_type(col_type)
        ]
        if not len(available_cols):
            return

        filtered_dataset = dataset[available_cols]
        if not isinstance(filtered_dataset, pd.DataFrame):
            return
        matrix = plot_correlation_matrix(df=filtered_dataset)
        return {
            "plot": matrix,
            "artifact_name": "correlation.html",
        }
