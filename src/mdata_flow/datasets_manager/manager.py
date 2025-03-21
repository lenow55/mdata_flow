import os
from functools import partial

from mlflow.client import MlflowClient
from mlflow.entities import Run
from mlflow.store.artifact.artifact_repository_registry import (
    _artifact_repository_registry,  # pyright: ignore[reportPrivateUsage]
)
from mlflow.store.artifact.optimized_s3_artifact_repo import (
    OptimizedS3ArtifactRepository,
)
from typing_extensions import Any

from mdata_flow.config import DatasetStoreSettings
from mdata_flow.datasets_manager.interfaces import DatasetVisitor, IDataset
from mdata_flow.datasets_manager.visitors import (
    ArtifactUploaderDatasetVisitor,
    CacheMoverDatasetVisitor,
    XXHDigestDatasetVisitor,
)
from mdata_flow.datasets_manager.visitors.scoped_abs_info_uploader import (
    ScopedABSUploaderVisitor,
)
from mdata_flow.types import NestedDict


def get_or_create_experiment(
    client: MlflowClient,
    experiment_name: str,
    artifact_location: str | None = None,
    tags: dict[str, Any] | None = None,
):
    if experiment := client.get_experiment_by_name(experiment_name):
        if isinstance(experiment.experiment_id, str):
            return experiment.experiment_id
        raise RuntimeError("Bad experiment_id type")
    else:
        return client.create_experiment(experiment_name, artifact_location, tags)


class DatasetManager:
    _saver: DatasetVisitor
    _upload_results: NestedDict[str | None] = {}
    _dataset_composite: IDataset | None = None

    def __init__(self, config: DatasetStoreSettings, saver: DatasetVisitor) -> None:
        self.config: DatasetStoreSettings = config
        self._experiment_id: str | None = None
        self._actual_run: Run | None = None
        self._client: MlflowClient = MlflowClient(tracking_uri=config.tracking_uri)
        self._saver = saver

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

    def register_datasets(self, dataset_composite: IDataset, run_name: str) -> bool:
        if not isinstance(self._experiment_id, str):
            raise RuntimeError("Run setup first")

        digest_v = XXHDigestDatasetVisitor()
        mover_v = CacheMoverDatasetVisitor(
            cache_folder=self.config.local_cache, store_run_name=run_name
        )
        uploader_v = ArtifactUploaderDatasetVisitor(
            client=self._client,
            cache_folder=self.config.local_cache,
            experiment_id=self._experiment_id,
            run_name=run_name,
        )

        dataset_composite.Accept(self._saver)
        dataset_composite.Accept(digest_v)
        dataset_composite.Accept(mover_v)
        dataset_composite.Accept(uploader_v)

        self._upload_results = uploader_v.get_results()
        self._dataset_composite = dataset_composite
        self._actual_run = uploader_v.get_run()

        if self._actual_run:
            return True
        else:
            return False

    def register_extra_uploaders(
        self,
        visitors: list[ScopedABSUploaderVisitor],
    ):
        """
        Универсальная функция для загрузки информации
        о датасетах, по типу графиков, превью, отчётов

        """
        if not isinstance(self._experiment_id, str):
            raise RuntimeError("Run setup first")

        if not isinstance(self._dataset_composite, IDataset):
            raise RuntimeError("Register datasets first")

        if not isinstance(self._actual_run, Run):
            raise RuntimeError("No actual_run")

        # генерируем и загружаем графики, превью, отчёты
        for visitor in visitors:
            visitor.client = self._client
            visitor.run = self._actual_run
            visitor.set_scope(self.get_results())
            self._dataset_composite.Accept(visitor)

    def finish_upload(self):
        if self._actual_run:
            self._client.set_terminated(self._actual_run.info.run_id, status="FINISHED")
            print("RUN FINISHED")
        else:
            print("No actual_run")

    def get_results(self):
        return self._upload_results
