from io import BufferedIOBase

import xxhash
from typing_extensions import override

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_visitor import (
    NestedDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.utils import FileResult


class XXHDigestDatasetVisitor(NestedDatasetVisitor[FileResult, FileResult]):
    def __init__(self) -> None:
        super().__init__()

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
    def _visit_pd_dataset(self, elem: PdDataset) -> FileResult:
        file_info = self._params_tmp_link.get(elem.name)
        if not file_info or isinstance(file_info, dict):
            raise RuntimeError(f"File was not saved, bad params {self._params}")
        digest = self._compute_xxhash(file_info.file_path)
        elem.digest = digest
        return file_info
