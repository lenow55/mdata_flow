from collections.abc import Generator
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pandas as pd
import pytest

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.utils import FileResult


def provide_dataset(
    name: str, file_type: str, set_digest: bool = True
) -> tuple[PdDataset, FileResult]:
    dataset = PdDataset(
        name=name, dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
    )
    if set_digest:
        dataset.digest = dataset.name
    return dataset, FileResult(
        file_path=NamedTemporaryFile(delete=False).name, file_type=file_type
    )


@pytest.fixture()
def cache_dir() -> Generator[str, None, None]:
    with TemporaryDirectory() as tempdir:
        yield tempdir
