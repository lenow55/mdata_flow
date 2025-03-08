import os
import shutil
from pathlib import Path

from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.file_name_validator import FileNameValidator


class CacheMoverDatasetVisitor(TypedDatasetVisitor):
    """
    Перемещает файлы датасетов в директорию кэша
    """

    # Результаты перемещения, заносятся все пути датасетов
    # решение загружать или нет принимает загрузчик
    _results: dict[str, str] = {}
    _current_ds_name: str = ""

    def __init__(self, cache_folder: str, store_run_name: str) -> None:
        if not FileNameValidator.is_valid(store_run_name):
            store_run_name = FileNameValidator.sanitize(store_run_name)
        self._store_path: Path = Path(cache_folder, store_run_name)
        if not os.path.exists(self._store_path):
            os.makedirs(self._store_path)

    @override
    def VisitPdDataset(self, elem: PdDataset):
        store_dataset_path = Path(self._store_path, f"{elem.digest}.{elem.file_type}")
        if os.path.exists(store_dataset_path) and os.path.samefile(
            elem.temp_path, store_dataset_path
        ):
            os.remove(elem.temp_path)
        else:
            shutil.move(elem.temp_path, store_dataset_path)
        self._results.update({self._current_ds_name: store_dataset_path.as_posix()})
        elem.file_path = store_dataset_path.as_posix()

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for value in elem.datasets:
            self._current_ds_name = value.name
            value.Accept(visitor=self)
