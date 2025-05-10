import os
import shutil
from pathlib import Path

from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_visitor import (
    NestedDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.utils import FileResult
from mdata_flow.file_name_validator import FileNameValidator


class CacheMoverDatasetVisitor(NestedDatasetVisitor[FileResult, Path]):
    """
    Перемещает файлы датасетов в директорию кэша
    """

    # Результаты перемещения, заносятся все пути датасетов
    # решение загружать или нет принимает загрузчик

    def __init__(self, cache_folder: str | Path, store_run_name: str) -> None:
        super().__init__()
        if not FileNameValidator.is_valid(store_run_name):
            store_run_name = FileNameValidator.sanitize(store_run_name)
        self._store_path: Path = Path(cache_folder, store_run_name)
        if not os.path.exists(self._store_path):
            os.makedirs(self._store_path)

    @override
    def _visit_pd_dataset(self, elem: PdDataset) -> Path:
        file_info = self._params_tmp_link.get(elem.name)
        if not file_info or isinstance(file_info, dict):
            raise RuntimeError(f"File was not saved, bad params {self._params}")

        store_dataset_path = Path(
            self._store_path, f"{elem.digest}.{file_info.file_type}"
        )

        if not os.path.exists(store_dataset_path):
            try:
                if os.path.samefile(file_info.file_path, store_dataset_path):
                    raise RuntimeError(
                        f"same files: {file_info.file_path} - {store_dataset_path}"
                    )
            except OSError:
                pass
            _ = shutil.move(file_info.file_path, store_dataset_path)
        elem.file_path = store_dataset_path
        return store_dataset_path
