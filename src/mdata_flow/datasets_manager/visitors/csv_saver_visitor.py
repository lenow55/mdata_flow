import tempfile

from pandas._typing import CompressionOptions
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor


class CSVSaverDatasetVisitor(TypedDatasetVisitor):
    """
    Сохраняет файлики CSV во временную директорию
    Результаты прям в объект датасета пишет
    Не ограничен уровень вложенности
    """

    def __init__(self, compression: CompressionOptions = "infer") -> None:
        super().__init__()
        self._compression: CompressionOptions = compression

    @override
    def VisitPdDataset(self, elem: PdDataset):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        df = elem.getDataset()
        _ = df.to_csv(temp_file, compression=self._compression)
        temp_file.flush()
        elem.temp_path = temp_file.name
        elem.file_type = "csv"
        if self._compression != "infer":
            if isinstance(self._compression, dict):
                elem.file_type = elem.file_type + f".{self._compression['method']}"
            else:
                elem.file_type = elem.file_type + f".{self._compression}"

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for value in elem.datasets:
            value.Accept(visitor=self)
