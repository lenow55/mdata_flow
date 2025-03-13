from io import BufferedIOBase
from unittest.mock import patch

import pandas as pd
import pytest

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.visitors import XXHDigestDatasetVisitor
from mdata_flow.types import NestedDict


def fake_xxhash(file: str | BufferedIOBase):
    return f"fake_hash_{file}" if isinstance(file, str) else "fake_hash_buffer"


dataset1 = PdDataset(
    name="file1.csv", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset1.temp_path = dataset1.name
dataset2 = PdDataset(
    name="file2.csv", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset2.temp_path = dataset2.name
dataset3 = PdDataset(
    name="file3.csv", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
)
dataset3.temp_path = dataset3.name

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
    "in_composite, expected_result",
    [
        pytest.param(
            GroupDataset(name="root", datasets=[dataset1, dataset2, dataset3]),
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
                        "file1.csv": "fake_hash_file1.csv",
                    },
                    "file1.csv": "fake_hash_file1.csv",
                    "file2.csv": "fake_hash_file2.csv",
                },
                "file3.csv": "fake_hash_file3.csv",
            },
            id="test_xxh_digest_visitor_flat",
        ),
    ],
)
def test_xxhdigest_nested(in_composite: GroupDataset, expected_result: NestedDict[str]):
    visitor = XXHDigestDatasetVisitor()

    with patch.object(
        XXHDigestDatasetVisitor, "_compute_xxhash", side_effect=fake_xxhash
    ):
        in_composite.Accept(visitor)
    res = visitor.get_results()
    assert res == expected_result
