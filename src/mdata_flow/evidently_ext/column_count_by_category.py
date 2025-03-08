from typing import Any

import pandas as pd
from evidently.base_metric import InputData, Metric, MetricResult
from evidently.core import IncludeOptions
from evidently.model.widget import BaseWidgetInfo
from evidently.renderers.base_renderer import MetricRenderer, default_renderer
from evidently.renderers.html_widgets import (
    TabData,
    header_text,
    table_data,
    widget_tabs,
)
from pydantic import BaseModel


class ColumnCountByCategory(BaseModel):
    categories: list[Any]
    counts: list[int]
    percents: list[float]


class ColumnCountByCategoryResult(MetricResult):
    column_name: str
    current: ColumnCountByCategory
    reference: ColumnCountByCategory | None

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        type_alias: str = "evidently:metric_result:ColumnCountByCategoryResult"


class ColumnCountByCategoryMetric(Metric[ColumnCountByCategoryResult]):
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        type_alias: str = "evidently:metric:ColumnCountByCategoryMetric"

    column_name: str
    round_c: int
    norm_column: str = "norm"

    def __init__(self, column_name: str, round_c: int) -> None:
        self.column_name = column_name
        self.round_c = round_c
        super().__init__()  ## pyright: ignore[reportUnknownMemberType]

    def calculate(self, data: InputData) -> ColumnCountByCategoryResult:
        # Вычисляем метрики для референсного датасета
        if data.reference_data is not None:
            reference_stat = pd.DataFrame(
                data=data.reference_data[self.column_name].value_counts(
                    normalize=False, dropna=False
                ),
            )
            reference_stat = reference_stat.assign(
                proportion=data.reference_data[self.column_name]
                .value_counts(normalize=True, dropna=False)
                .apply(lambda x: round(x * 100, self.round_c))
            )
            reference_stat = reference_stat.sort_values(
                by="proportion", ascending=False
            )
            reference_categories = reference_stat.index.tolist()
            reference_counts = reference_stat["count"].values.tolist()
            reference_percents = reference_stat["proportion"].values.tolist()
            reference = ColumnCountByCategory(
                categories=reference_categories,
                counts=reference_counts,
                percents=reference_percents,
            )
        else:
            reference = None

        # Вычисляем метрики для текущего датасета
        current_stat = pd.DataFrame(
            data=data.current_data[self.column_name].value_counts(
                normalize=False, dropna=False
            ),
        )
        current_stat = current_stat.assign(
            proportion=data.current_data[self.column_name]
            .value_counts(normalize=True, dropna=False)
            .apply(lambda x: round(x * 100, self.round_c))
        )
        current_stat = current_stat.sort_values(by="proportion", ascending=False)

        current_categories = current_stat.index.tolist()
        current_counts = current_stat["count"].values.tolist()
        current_percents = current_stat["proportion"].values.tolist()

        current = ColumnCountByCategory(
            categories=current_categories,
            counts=current_counts,
            percents=current_percents,
        )

        return ColumnCountByCategoryResult(  # pyright: ignore[reportCallIssue]
            column_name=self.column_name,  # pyright: ignore[reportCallIssue]
            current=current,  # pyright: ignore[reportCallIssue]
            reference=reference,  # pyright: ignore[reportCallIssue]
        )


@default_renderer(wrap_type=ColumnCountByCategoryMetric)
class ColumnCountByCategoryMetricRenderer(MetricRenderer[ColumnCountByCategoryMetric]):
    def _get_table_stat(
        self, dataset_name: str, stats: ColumnCountByCategory
    ) -> BaseWidgetInfo:
        matched_stat = zip(stats.categories, stats.counts, stats.percents)
        matched_stat_headers = ["Name", "Count", "Percent"]
        table_tab = table_data(
            title="",
            column_names=matched_stat_headers,
            data=matched_stat,
        )
        return widget_tabs(
            title=f"{dataset_name.capitalize()} dataset",
            tabs=[
                TabData(title="Table", widget=table_tab),
            ],
        )

    def render_json(
        self,
        obj: ColumnCountByCategoryMetric,
        include_render: bool = False,
        include: IncludeOptions | None = None,
        exclude: IncludeOptions | None = None,
    ) -> dict[Any, Any]:
        result = obj.get_result().get_dict(include_render, include, exclude)
        return result

    def render_html(self, obj: ColumnCountByCategoryMetric) -> list[BaseWidgetInfo]:
        metric_result = obj.get_result()
        result = [
            header_text(label=f"{obj.column_name}"),
            self._get_table_stat(dataset_name="current", stats=metric_result.current),
        ]

        if metric_result.reference is not None:
            result.append(
                self._get_table_stat(
                    dataset_name="reference", stats=metric_result.reference
                )
            )

        return result
