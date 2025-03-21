from io import BufferedIOBase
from typing_extensions import override

import xxhash

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.nested_results_visitor import (
    NestedResultsDatasetVisitor,
)


class XXHDigestDatasetVisitor(NestedResultsDatasetVisitor[str]):
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
    def _visit_pd_dataset(self, elem: PdDataset) -> str:
        digest = self._compute_xxhash(elem.temp_path)
        elem.digest = digest
        return digest
