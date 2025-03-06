from typing import TypedDict
from typing_extensions import Any


class FigureArtifact(TypedDict):
    plot: Any
    artifact_name: str
