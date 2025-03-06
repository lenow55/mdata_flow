from io import BufferedIOBase

import xxhash
from typing_extensions import override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor


class XXHDigestDatasetVisitor(TypedDatasetVisitor):
    _results: dict[str, str] = {}
    _current_type: str | None = None

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
        if not self._current_type:
            return
        self._results.update({self._current_type: digest})

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        for ds_type, value in elem.datasets.items():
            self._current_type = ds_type
            value.Accept(visitor=self)
