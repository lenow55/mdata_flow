import tempfile

from pandas._typing import CompressionOptions
from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_visitor import (
    NestedDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.utils import FileResult


class CSVSaverDatasetVisitor(NestedDatasetVisitor[None, FileResult]):
    """
    Сохраняет файлики CSV во временную директорию
    Не ограничен уровень вложенности
    """

    def __init__(
        self, compression: CompressionOptions = "infer", file_extension: str = "csv"
    ) -> None:
        super().__init__()
        self._compression: CompressionOptions = compression
        self._extension: str = file_extension

    @override
    def _visit_pd_dataset(self, elem: PdDataset) -> FileResult:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        df = elem.getDataset()
        _ = df.to_csv(temp_file, compression=self._compression)
        temp_file.flush()
        result = FileResult(file_path=temp_file.name, file_type=self._extension)
        return result
