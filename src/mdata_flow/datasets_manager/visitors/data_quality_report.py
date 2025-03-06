from typing import final

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from typing_extensions import override

from mdata_flow.datasets_manager.visitors.evidently_report_visitor import (
    EvidentlyReportVisitor,
)


class DataQualityReportVisitor(EvidentlyReportVisitor):
    """
    Рассчитывает отчёт количества по категориям
    """

    @final
    @override
    def _pandas_build_report(self) -> Report | None:
        report = Report(metrics=[DataQualityPreset()])
        report.name = "data_quality"

        return report
