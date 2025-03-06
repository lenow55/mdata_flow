from typing import final
from mlflow.types.schema import DataType
from typing_extensions import Any, override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.figure_visitor import FigureVisitor
from mdata_flow.datasets_manager.visitors.utils import FigureArtifact
from mdata_flow.file_name_validator import FileNameValidator
from mdata_flow.plotly_ext.func_library import plot_density_diagram


class PlotlyDensityVisitor(FigureVisitor):
    """
    Рассчитывает график плотности числового значения
    по категориальным значениям
    """

    def __init__(
        self,
        categorical_col: str,
        numeric_col: str,
        labels_map: dict[Any, str] | None = None,
        plot_size: tuple[int, int] = (800, 600),
    ) -> None:
        super().__init__(plot_size=plot_size)
        self._categorical_column: str = categorical_col
        self._numeric_column: str = numeric_col
        self._labels_map = labels_map

    def _check_type(self, data_type: Any) -> bool:
        if not isinstance(data_type, DataType):
            return False

        return data_type in [
            DataType.integer,
            DataType.double,
            DataType.long,
            DataType.float,
        ]

    @final
    @override
    def _pandas_plot_figure(self, elem: PdDataset) -> FigureArtifact | None:
        num_data_type = elem.schema.input_types_dict()[self._numeric_column]
        if not self._check_type(num_data_type):
            print(
                f"Warning can't use col {self._numeric_column} due to data type: {num_data_type}"
            )
            return None

        dataset = elem.getDataset()

        density_plot = plot_density_diagram(
            df=dataset,
            numeric_col=self._numeric_column,
            categorical_col=self._categorical_column,
            labels_map=self._labels_map,
            plot_size=self._plot_size,
        )

        artifact_name = (
            f"density_plot_{self._numeric_column}_by_{self._categorical_column}.html"
        )
        if not FileNameValidator.is_valid(artifact_name):
            artifact_name = FileNameValidator.sanitize(artifact_name)

        return {
            "plot": density_plot,
            "artifact_name": artifact_name,
        }
