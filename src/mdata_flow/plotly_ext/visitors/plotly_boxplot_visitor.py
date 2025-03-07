from typing import final
from mlflow.types.schema import DataType
from typing_extensions import Any, override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.figure_visitor import FigureVisitor
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact
from mdata_flow.file_name_validator import FileNameValidator
from mdata_flow.plotly_ext.func_library import plot_box_diagram


@final
class PlotlyBoxplotVisitor(FigureVisitor):
    """
    Рассчитывает матрицу корелляций для датасетов
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        plot_size: tuple[int, int] = (800, 600),
    ) -> None:
        super().__init__(plot_size=plot_size)
        self._x_col: str = x_col
        self._y_col: str = y_col

    def _check_type(self, data_type: Any) -> bool:
        if not isinstance(data_type, DataType):
            return False

        return data_type in [
            DataType.integer,
            DataType.double,
            DataType.long,
            DataType.float,
        ]

    @override
    def _pandas_plot_figure(self, elem: PdDataset) -> FigureArtifact | None:
        y_data_type = elem.schema.input_types_dict()[self._y_col]
        if not self._check_type(y_data_type):
            print(
                f"Warning can't use col {self._y_col} due to data type: {y_data_type}"
            )
            return None

        dataset = elem.getDataset()

        box_plot = plot_box_diagram(
            df=dataset, x_col=self._x_col, y_col=self._y_col, plot_size=self._plot_size
        )

        artifact_name = f"box_plot_{self._y_col}_on_{self._x_col}.html"
        if not FileNameValidator.is_valid(artifact_name):
            artifact_name = FileNameValidator.sanitize(artifact_name)

        return {
            "plot": box_plot,
            "artifact_name": artifact_name,
        }
