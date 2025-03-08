from collections.abc import Iterator
from contextlib import contextmanager
from io import BufferedIOBase

import xxhash
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.types import NestedDict


class XXHDigestDatasetVisitor(TypedDatasetVisitor):
    _results: NestedDict
    # ссылка на текущий корень обработки
    _results_tmp_link: NestedDict
    # список ключей текущего уровня
    _current_ds_key_path: list[str]

    def __init__(self) -> None:
        super().__init__()
        self._results = {}
        self._results_tmp_link = self._results
        self._current_ds_key_path = []

    def get_results(self):
        return self._results

    @staticmethod
    def _compute_xxhash(file: str | BufferedIOBase):
        """Вычислить xxh хэш для файла."""
        str_hash = xxhash.xxh3_64()
        if isinstance(file, str):
            with open(file, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    str_hash.update(byte_block)
        else:
            for byte_block in iter(lambda: file.read(8192), b""):
                str_hash.update(byte_block)

        return str_hash.hexdigest()

    @override
    def VisitPdDataset(self, elem: PdDataset):
        digest = self._compute_xxhash(elem.temp_path)
        elem.digest = digest

        if not len(self._current_ds_key_path):
            raise RuntimeError("Run without GroupDataset")

        # забираем текущий ключ из списка и по нему назначаем
        # результат
        try:
            key = self._current_ds_key_path[-1]
            self._results_tmp_link.update({key: digest})
        except KeyError as e:
            e.add_note("Bad keys list")
            raise

    @contextmanager
    def _manage_path(self) -> Iterator[None]:
        backup_tmp_link = self._results_tmp_link
        if len(self._current_ds_key_path):
            # если путь не пустой, значит вызваны из группы
            self._results_tmp_link.update({self._current_ds_key_path[-1]: {}})
            tmp_link = self._results_tmp_link[self._current_ds_key_path[-1]]
            if not isinstance(tmp_link, dict):
                raise RuntimeError("Bad tmp_link XXHDigestDatasetVisitor")

            # переносим ссылку на новую вложенность
            self._results_tmp_link = tmp_link

        yield

        if len(self._current_ds_key_path):
            self._results_tmp_link = backup_tmp_link

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        with self._manage_path():
            for value in elem.datasets:
                # добавляем ключ датасета, в который заходить будем
                self._current_ds_key_path.append(value.name)
                value.Accept(visitor=self)
                # извлекаем ключ, не нужен
                _ = self._current_ds_key_path.pop()
