from collections.abc import Generator
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, TemporaryFile

import pandas as pd
import pytest

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors import CacheMoverDatasetVisitor
from mdata_flow.types import NestedDict


dataset1 = PdDataset(
    name="file1", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset1.temp_path = NamedTemporaryFile(delete=False).name
dataset1.digest = dataset1.name
dataset1.file_type = "csv"
dataset2 = PdDataset(
    name="file2", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset2.temp_path = NamedTemporaryFile(delete=False).name
dataset2.digest = dataset2.name
dataset2.file_type = "csv"
dataset3 = PdDataset(
    name="file3", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset3.temp_path = NamedTemporaryFile(delete=False).name
dataset3.file_type = "csv"
dataset3.digest = dataset3.name

subgroup2 = GroupDataset(name="subgroup2", datasets=[dataset1])
subgroup = GroupDataset(
    name="subgroup",
    datasets=[
        subgroup2,
        dataset1,
        dataset2,
    ],
)
root_group = GroupDataset(
    name="root",
    datasets=[GroupDataset(name="subgroup", datasets=[dataset1, dataset2]), dataset3],
)
root2_group = GroupDataset(
    name="root",
    datasets=[subgroup, dataset3],
)


@pytest.fixture()
def cache_dir() -> Generator[str, None, None]:
    with TemporaryDirectory() as tempdir:
        yield tempdir


# FIXME: надо перенести инициализацию датасетов в другое место
# чтобы временные файлы создавались каждый раз по новому


@pytest.mark.parametrize(
    "in_composite, expected_result, expected_dir_content, run_name",
    [
        pytest.param(
            GroupDataset(name="root", datasets=[dataset1, dataset2, dataset3]),
            {
                "file1": "test_run1\\file1.csv",
                "file2": "test_run1\\file2.csv",
                "file3": "test_run1\\file3.csv",
            },
            [
                "file1.csv",
                "file2.csv",
                "file3.csv",
            ],
            "test_run1",
            id="test_cache_mover_visitor_flat",
        ),
        pytest.param(
            root2_group,
            {
                "subgroup": {
                    "subgroup2": {
                        "file1": "test_run2/file1.csv",
                    },
                    "file1": "test_run2/file1.csv",
                    "file2": "test_run2/file2.csv",
                },
                "file3": "test_run2/file3.csv",
            },
            [
                "file1.csv",
                "file2.csv",
                "file3.csv",
            ],
            "test_run2",
            id="test_cache_mover_visitor_nested",
        ),
    ],
)
def test_cache_mover_visitor(
    in_composite: GroupDataset,
    expected_result: NestedDict[str],
    expected_dir_content: list[str],
    run_name: str,
    cache_dir: str,
):
    visitor = CacheMoverDatasetVisitor(cache_folder=cache_dir, store_run_name=run_name)

    in_composite.Accept(visitor)

    res = visitor.get_results()

    def fixer(nest: NestedDict[str]):
        for key, value in nest.items():
            if isinstance(value, dict):
                fixer(value)
            else:
                nest[key] = Path(os.path.join(cache_dir, value)).as_posix()

    fixer(expected_result)

    list_runs = os.listdir(cache_dir)
    list_files = os.listdir(os.path.join(cache_dir, run_name))
    assert len(list_runs) == 1
    assert list_runs[0] == run_name
    assert list_files == expected_dir_content
    assert res == expected_result
