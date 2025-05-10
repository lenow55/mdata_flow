import tempfile

from pandas._typing import (
    CompressionOptions,  # pyright: ignore[reportPrivateImportUsage]
)
from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_visitor import (
    NestedDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.utils import FileResult


class CSVSaverDatasetVisitor(NestedDatasetVisitor[None, FileResult]):
    """
    Сохраняет файлики CSV во временную директорию
    Результаты прям в объект датасета пишет
    Не ограничен уровень вложенности
    """

    def __init__(self, compression: CompressionOptions = "infer") -> None:
        super().__init__()
        self._compression: CompressionOptions = compression

    @override
    def _visit_pd_dataset(self, elem: PdDataset) -> FileResult:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        df = elem.getDataset()
        _ = df.to_csv(temp_file, compression=self._compression)
        temp_file.flush()
        file_type = "csv"
        if self._compression != "infer":
            if isinstance(self._compression, dict):
                file_type = file_type + f".{self._compression['method']}"
            else:
                file_type = file_type + f".{self._compression}"
        result = FileResult(file_path=temp_file.name, file_type=file_type)
        return result
