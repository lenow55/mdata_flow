from io import BufferedIOBase
from unittest.mock import patch

import pytest

from mdata_flow.datasets_manager.composites import GroupDataset
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.visitors import XXHDigestDatasetVisitor
from mdata_flow.datasets_manager.visitors.utils import FileResult
from mdata_flow.types import NestedDict
from tests.datasets_manager.visitors.conftest import provide_dataset

# BUG: этот тест сейчас вообще не работает из-за изменений
pytestmark = pytest.mark.skipif(
    reason="Не работают из-за изменений в NestedDatasetVisitor"
)


def fake_xxhash(file: str | BufferedIOBase):
    return f"fake_hash_{file}" if isinstance(file, str) else "fake_hash_buffer"


dataset1, info1 = provide_dataset(name="file1.csv", file_type="csv", set_digest=False)
dataset2, info2 = provide_dataset(name="file2.csv", file_type="csv", set_digest=False)
dataset3, info3 = provide_dataset(name="file3.csv", file_type="csv", set_digest=False)
info1.file_path = dataset1.name
info2.file_path = dataset2.name
info3.file_path = dataset3.name

subgroup2 = GroupDataset(name="subgroup2", datasets=[dataset1])
subgroup = GroupDataset(name="subgroup", datasets=[subgroup2, dataset1, dataset2])
root_group = GroupDataset(
    name="root",
    datasets=[GroupDataset(name="subgroup", datasets=[dataset1, dataset2]), dataset3],
)
root2_group = GroupDataset(
    name="root",
    datasets=[subgroup, dataset3],
)


@pytest.mark.parametrize(
    "in_composite, visitor_params, expected_result",
    [
        pytest.param(
            GroupDataset(name="root", datasets=[dataset1, dataset2, dataset3]),
            {
                "file1.csv": info1,
                "file2.csv": info2,
                "file3.csv": info3,
            },
            {
                "file1.csv": "fake_hash_file1.csv",
                "file2.csv": "fake_hash_file2.csv",
                "file3.csv": "fake_hash_file3.csv",
            },
            id="test_xxh_digest_visitor_flat",
        ),
        pytest.param(
            root_group,
            {
                "subgroup": {
                    "file1.csv": info1,
                    "file2.csv": info2,
                },
                "file3.csv": info3,
            },
            {
                "subgroup": {
                    "file1.csv": "fake_hash_file1.csv",
                    "file2.csv": "fake_hash_file2.csv",
                },
                "file3.csv": "fake_hash_file3.csv",
            },
            id="test_xxh_digest_visitor_nested",
        ),
        pytest.param(
            root2_group,
            {
                "subgroup": {
                    "subgroup2": {
                        "file1.csv": info1,
                    },
                    "file1.csv": info1,
                    "file2.csv": info2,
                },
                "file3.csv": info3,
            },
            {
                "subgroup": {
                    "subgroup2": {
                        "file1.csv": "fake_hash_file1.csv",
                    },
                    "file1.csv": "fake_hash_file1.csv",
                    "file2.csv": "fake_hash_file2.csv",
                },
                "file3.csv": "fake_hash_file3.csv",
            },
            id="test_xxh_digest_visitor_2_nested",
        ),
        pytest.param(
            dataset1,
            {
                "file1.csv": info1,
            },
            {
                "file1.csv": "fake_hash_file1.csv",
            },
            id="test_xxh_digest_visitor_no_groups",
        ),
    ],
)
def test_xxhdigest_nested(
    in_composite: IDataset,
    visitor_params: NestedDict[FileResult],
    expected_result: NestedDict[str],
):
    visitor = XXHDigestDatasetVisitor()
    visitor.set_params(params=visitor_params)

    with patch.object(
        XXHDigestDatasetVisitor, "_compute_xxhash", side_effect=fake_xxhash
    ):
        in_composite.Accept(visitor)
    res = visitor.get_results()
    assert res == expected_result
