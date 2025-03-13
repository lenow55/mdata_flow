import os
import shutil
from pathlib import Path

from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_results_visitor import (
    NestedResultsDatasetVisitor,
)
from mdata_flow.file_name_validator import FileNameValidator


class CacheMoverDatasetVisitor(NestedResultsDatasetVisitor[str]):
    """
    Перемещает файлы датасетов в директорию кэша
    """

    # Результаты перемещения, заносятся все пути датасетов
    # решение загружать или нет принимает загрузчик

    def __init__(self, cache_folder: str, store_run_name: str) -> None:
        super().__init__()
        if not FileNameValidator.is_valid(store_run_name):
            store_run_name = FileNameValidator.sanitize(store_run_name)
        self._store_path: Path = Path(cache_folder, store_run_name)
        if not os.path.exists(self._store_path):
            os.makedirs(self._store_path)

    @override
    def _visit_pd_dataset(self, elem: PdDataset) -> str:
        store_dataset_path = Path(self._store_path, f"{elem.digest}.{elem.file_type}")
        # INFO: Проверка на samefile не работает, так как файл
        # после перемещения уже недоступен по temp_path
        if not os.path.exists(store_dataset_path):
            shutil.move(elem.temp_path, store_dataset_path)
        elem.file_path = store_dataset_path.as_posix()
        return store_dataset_path.as_posix()
