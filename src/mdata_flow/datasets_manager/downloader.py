import os
from functools import partial

from mlflow.client import MlflowClient
from mlflow.data import get_source
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.entities import Run
from mlflow.store.artifact.artifact_repository_registry import (
    _artifact_repository_registry,  # pyright: ignore[reportPrivateUsage]
)
from mlflow.store.artifact.optimized_s3_artifact_repo import (
    OptimizedS3ArtifactRepository,
)

from mdata_flow.config import DatasetStoreSettings
from mdata_flow.datasets_manager.utils import get_or_create_experiment


class DatasetDownloader:
    def __init__(
        self,
        config: DatasetStoreSettings,
    ) -> None:
        self.config: DatasetStoreSettings = config
        self._experiment_id: str | None = None
        self._actual_run: Run | None = None
        self._client: MlflowClient = MlflowClient(tracking_uri=config.tracking_uri)

    def setup(self):
        s3withCreds = partial(
            OptimizedS3ArtifactRepository,
            access_key_id=self.config.access_key_id,
            secret_access_key=self.config.secret_access_key.get_secret_value(),
            s3_endpoint_url=self.config.s3_endpoint_url,
        )
        _artifact_repository_registry.register("s3", s3withCreds)

        self._experiment_id = get_or_create_experiment(
            self._client,
            self.config.data_experiment.name,
            self.config.data_experiment.artifact_path,
        )
        if not os.path.exists(self.config.local_cache):
            os.mkdir(self.config.local_cache)

    def download(
        self, run_name: str, run_version: int = -1, dataset_name: str | None = None
    ) -> dict[str, str]:
        if not self._experiment_id:
            raise RuntimeError("Run setup first")

        filter_string = (
            f'attributes.run_name = "{run_name}" AND attributes.status = "FINISHED"'
        )
        if run_version >= 0:
            filter_string += f' AND tags.version = "{run_version}"'

        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
            order_by=["tags.version DESC"],
        )

        try:
            run = runs[0]
        except IndexError:
            return {}

        result: dict[str, str] = {}
        for dataset_input in run.inputs.dataset_inputs:
            if dataset_name and dataset_name != dataset_input.dataset.name:
                continue
            source = get_source(dataset_input)
            if not isinstance(source, FileSystemDatasetSource):
                continue
            if not isinstance(source.uri, str):
                continue
            file_name: str = os.path.basename(source.uri)
            cache_file = os.path.join(self.config.local_cache, run_name, file_name)
            if not os.path.exists(cache_file):
                local_dataset_source = source.load(dst_path=os.path.dirname(cache_file))
                cache_file = local_dataset_source
            result.update({dataset_input.dataset.name: cache_file})
        return result
