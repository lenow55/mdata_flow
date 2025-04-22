from dataclasses import dataclass
from typing import TypedDict
from typing_extensions import Any


class FigureArtifact(TypedDict):
    plot: Any
    artifact_name: str


@dataclass
class FileResult:
    """
    Тут может храниться имя или временного
    или обычного файла
    """

    file_path: str
    file_type: str
