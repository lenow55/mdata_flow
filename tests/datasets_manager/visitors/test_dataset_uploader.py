import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from mlflow.client import MlflowClient
from mlflow.entities import DatasetInput, Run

from mdata_flow.datasets_manager.composites import PdDataset
from mdata_flow.datasets_manager.visitors.dataset_uploader_mlflow_visitor import (
    ArtifactUploaderDatasetVisitor,
)


class TestArtifactUploaderDatasetVisitor:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=MlflowClient)
        client.search_runs.return_value = []
        client.create_run.return_value = MagicMock(
            spec=Run,
            info=MagicMock(
                run_id="test_run_id", artifact_uri="s3://test-bucket/artifacts"
            ),
        )
        return client

    @pytest.fixture
    def mock_context_registry(self):
        with patch(
            "mdata_flow.datasets_manager.visitors.dataset_uploader_mlflow_visitor.context_registry"
        ) as registry:
            registry.resolve_tags.return_value = {"version": "1", "resolved": True}
            yield registry

    @pytest.fixture
    def visitor(self, mock_client: MlflowClient) -> ArtifactUploaderDatasetVisitor:
        return ArtifactUploaderDatasetVisitor(
            client=mock_client,
            cache_folder="/tmp/cache",
            experiment_id="test_exp_id",
            run_name="test_run",
        )

    def test_get_new_version_success(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        mock_run = MagicMock()
        mock_run.data.tags = {"version": "5"}
        mock_client.search_runs.return_value = [mock_run]

        version = visitor.get_new_version("test_run")
        assert isinstance(version, int)
        assert version == 6

    def test_get_new_version_failure(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        mock_client.search_runs.return_value = []
        version = visitor.get_new_version("test_run")
        assert version == 0

    def test_check_need_update_found(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        mock_run = MagicMock()
        mock_dataset_input = MagicMock(spec=DatasetInput)
        mock_dataset_input.dataset.digest = "test_digest"
        mock_run.inputs.dataset_inputs = [mock_dataset_input]
        mock_client.search_runs.return_value = [mock_run]

        assert visitor.check_need_update("test_digest") == False

    def test_check_need_update_not_found(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        mock_run = MagicMock()
        mock_run.inputs.dataset_inputs = []
        mock_client.search_runs.return_value = [mock_run]

        assert visitor.check_need_update("new_digest") == True

    def test_get_or_create_run_types(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        visitor._run = None
        run = visitor._get_or_create_run()

        assert isinstance(run.info.run_id, str)
        assert isinstance(run.info.artifact_uri, str)
        call_args = mock_client.create_run.call_args

        assert call_args[1]["experiment_id"] == "test_exp_id"
        tags = call_args[1]["tags"]
        assert tags.get("version") == "0"
        assert tags.get("mlflow.runName") == "test_run"

    def test_visit_pd_dataset(
        self, visitor: ArtifactUploaderDatasetVisitor, mock_client: MagicMock
    ):
        mock_run = MagicMock(spec=Run)
        mock_run.info.artifact_uri = "artifact_uri"
        mock_run.info.run_id = "test_run_id"
        visitor._get_or_create_run = MagicMock(return_value=mock_run)

        pd_dataset = PdDataset(
            name="test_data", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
        )
        pd_dataset.digest = "test_digest"
        pd_dataset.file_path = "test_digest.csv"

        visitor.Visit(pd_dataset)

        result = visitor.get_results()

        assert isinstance(result, dict)
        expected_result = {
            pd_dataset.name: os.path.join(
                "artifact_uri", "datasets", os.path.basename("test_digest.csv")
            )
        }
        assert result == expected_result

        # Existing assertions for other functionality
        mock_client.log_artifact.assert_called_once_with(
            run_id="test_run_id",
            local_path=str(pd_dataset.file_path),
            artifact_path="datasets",
        )
        mock_client.log_inputs.assert_called_once()

    def test_dataset_path_property(self, visitor: ArtifactUploaderDatasetVisitor):
        visitor._current_ds_key_path = ["group1", "dataset1"]
        assert visitor._dataset_path == "datasets/group1"
