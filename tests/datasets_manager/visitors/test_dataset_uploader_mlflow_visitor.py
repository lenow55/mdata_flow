from unittest.mock import MagicMock
import pytest

from mdata_flow.datasets_manager.visitors import ArtifactUploaderDatasetVisitor


class TestDatasetArtifactUploader:
    client = MagicMock()

    @pytest.mark.parametrize(
        "keys_list, exp_result",
        [
            pytest.param(
                [],
                None,
                id="test__dataset_path_empty",
            ),
            pytest.param(
                ["file1"],
                "datasets",
                id="test__dataset_path_one",
            ),
        ],
    )
    def test_no_update(self, keys_list: list[str], exp_result: str):
        visitor = ArtifactUploaderDatasetVisitor(self.client, "", "", "")
        visitor.check_need_update = MagicMock(return_value=True)
