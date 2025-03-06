import typing

from mdata_flow.lazy_loader import LazyLoader

if typing.TYPE_CHECKING:
    # Plotly
    from mdata_flow.plotly_ext.visitors.plotly_boxplot_visitor import (
        PlotlyBoxplotVisitor,
    )
    from mdata_flow.plotly_ext.visitors.plotly_corr_visitor import (
        PlotlyCorrVisitor,
    )
    from mdata_flow.plotly_ext.visitors.plotly_density_visitor import (
        PlotlyDensityVisitor,
    )
else:
    # Plotly
    PlotlyBoxplotVisitor = LazyLoader(  # pyright: ignore[reportUnreachable]
        "mdata_flow.plotly_ext.visitors.plotly_boxplot_visitor"
    )
    PlotlyCorrVisitor = LazyLoader("mdata_flow.plotly_ext.visitors.plotly_corr_visitor")
    PlotlyDensityVisitor = LazyLoader(
        "mdata_flow.plotly_ext.visitors.plotly_density_visitor"
    )

__all__ = [
    "PlotlyCorrVisitor",
    "PlotlyBoxplotVisitor",
    "PlotlyDensityVisitor",
]
