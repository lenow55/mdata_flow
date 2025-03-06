import typing
from mdata_flow.lazy_loader import LazyLoader
from mdata_flow.datasets_manager.visitors.cache_mover_visitor import (
    CacheMoverDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.count_by_category_report import (
    CountByCategoryReportVisitor,
)
from mdata_flow.datasets_manager.visitors.csv_saver_visitor import (
    CSVSaverDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.data_quality_report import (
    DataQualityReportVisitor,
)
from mdata_flow.datasets_manager.visitors.dataset_uploader_mlflow_visitor import (
    ArtifactUploaderDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.evidently_report_visitor import (
    EvidentlyReportVisitor,
)
from mdata_flow.datasets_manager.visitors.figure_visitor import FigureVisitor
from mdata_flow.datasets_manager.visitors.figures_uploader_mlflow_visitor import (
    FiguresUploaderDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.preview_uploader_visitor import (
    PreviewUploaderVisitor,
)
from mdata_flow.datasets_manager.visitors.xxh_digest_visitor import (
    XXHDigestDatasetVisitor,
)

if typing.TYPE_CHECKING:
    from mdata_flow.datasets_manager.visitors.plotly.plotly_boxplot_visitor import (
        PlotlyBoxplotVisitor,
    )
    from mdata_flow.datasets_manager.visitors.plotly.plotly_corr_visitor import (
        PlotlyCorrVisitor,
    )
    from mdata_flow.datasets_manager.visitors.plotly.plotly_density_visitor import (
        PlotlyDensityVisitor,
    )
else:
    pass
    faster_whisper = LazyLoader("faster_whisper")
    fw_transcribe = LazyLoader("faster_whisper.transcribe")

__all__ = [
    "CSVSaverDatasetVisitor",
    "XXHDigestDatasetVisitor",
    "CacheMoverDatasetVisitor",
    "PlotlyCorrVisitor",
    "PlotlyBoxplotVisitor",
    "ArtifactUploaderDatasetVisitor",
    "FiguresUploaderDatasetVisitor",
    "FigureVisitor",
    "PreviewUploaderVisitor",
    "PlotlyDensityVisitor",
    "CountByCategoryReportVisitor",
    "DataQualityReportVisitor",
    "EvidentlyReportVisitor",
]
