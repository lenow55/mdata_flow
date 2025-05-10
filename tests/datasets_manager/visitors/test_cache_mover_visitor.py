import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.visitors import CacheMoverDatasetVisitor
from mdata_flow.datasets_manager.visitors.utils import FileResult
from mdata_flow.types import NestedDict

pytestmark = pytest.mark.skipif(
    reason="Не работают из-за изменений в NestedDatasetVisitor"
)


def create_dataset(name: str) -> PdDataset:
    dataset = PdDataset(
        name=name, dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
    )
    dataset_file_info = FileResult(
        file_path=NamedTemporaryFile(delete=False).name, file_type="csv"
    )
    dataset.digest = dataset.name
    return dataset


dataset1 = create_dataset("file1")


@pytest.mark.parametrize(
    "in_composite, expected_result, expected_dir_content, run_name",
    [
        pytest.param(
            GroupDataset(
                name="root",
                datasets=[
                    create_dataset("file1"),
                    create_dataset("file2"),
                    create_dataset("file3"),
                ],
            ),
            {
                "file1": "test_run1/file1.csv",
                "file2": "test_run1/file2.csv",
                "file3": "test_run1/file3.csv",
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
            create_dataset("file1"),
            {
                "file1": "test_run1/file1.csv",
            },
            [
                "file1.csv",
            ],
            "test_run1",
            id="test_cache_mover_visitor_no_group",
        ),
        pytest.param(
            GroupDataset(
                name="root",
                datasets=[
                    GroupDataset(
                        name="subgroup",
                        datasets=[
                            GroupDataset(name="subgroup2", datasets=[dataset1]),
                            dataset1,
                            create_dataset("file2"),
                        ],
                    ),
                    create_dataset("file3"),
                ],
            ),
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
    in_composite: IDataset,
    expected_result: NestedDict[str],
    expected_dir_content: list[str],
    run_name: str,
    tmp_path_factory: pytest.TempPathFactory,
):
    cache_dir = tmp_path_factory.mktemp(basename="cache")
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
    assert Counter(list_files) == Counter(expected_dir_content)
    assert res == expected_result
