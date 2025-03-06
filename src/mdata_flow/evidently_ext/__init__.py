import typing

from mdata_flow.lazy_loader import LazyLoader

if typing.TYPE_CHECKING:
    # Evidently
    from mdata_flow.evidently_ext.visitors.count_by_category_report import (
        CountByCategoryReportVisitor,
    )
    from mdata_flow.evidently_ext.visitors.data_quality_report import (
        DataQualityReportVisitor,
    )
    from mdata_flow.evidently_ext.visitors.evidently_abs_report_visitor import (
        EvidentlyReportVisitor,
    )
else:
    # Evidently
    EvidentlyReportVisitor = LazyLoader(  # pyright: ignore[reportUnreachable]
        "mdata_flow.evidently_ext.visitors.evidently_abs_report_visitor"
    )
    CountByCategoryReportVisitor = LazyLoader(
        "mdata_flow.evidently_ext.visitors.count_by_category_report"
    )
    DataQualityReportVisitor = LazyLoader(
        "mdata_flow.evidently_ext.visitors.data_quality_report"
    )

__all__ = [
    "CountByCategoryReportVisitor",
    "DataQualityReportVisitor",
    "EvidentlyReportVisitor",
]
