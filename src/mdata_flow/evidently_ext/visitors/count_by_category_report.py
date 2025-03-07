from typing import final

from evidently.report import Report
from typing_extensions import override

from mdata_flow.evidently_ext.column_count_by_category import (
    ColumnCountByCategoryMetric,
)
from mdata_flow.evidently_ext.visitors.evidently_abs_report_visitor import (
    EvidentlyReportVisitor,
)


class CountByCategoryReportVisitor(EvidentlyReportVisitor):
    """
    Рассчитывает отчёт количества по категориям
    """

    @final
    @override
    def _pandas_build_report(self) -> Report | None:
        if not self._column_maping.categorical_features:
            print("Can't generate report by empty cat cols")
            return None

        if len(self._column_maping.categorical_features) == 0:
            print("Can't generate report by empty cat cols")
            return None

        report = Report(  # pyright: ignore[reportArgumentType]
            metrics=[
                ColumnCountByCategoryMetric(column_name=name, round_c=2)
                for name in self._column_maping.categorical_features
            ],
            name="column_count_by_category",
        )

        return report
