import os
from pathlib import Path

from mlflow.client import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT, MLFLOW_RUN_NAME
from typing_extensions import Any, override

from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.context import DsContext
from mdata_flow.datasets_manager.visitors.typed_abs_visitor import TypedDatasetVisitor
from mdata_flow.file_name_validator import FileNameValidator


class ArtifactUploaderDatasetVisitor(TypedDatasetVisitor):
    """
    Загружает файлы датасетов на mlflow s3
    """

    _results: dict[str, str] = {}
    _run: Run | None = None

    _experiment_id: str
    _run_name: str
    _client: MlflowClient
    _store_path: Path

    _root_artifact_path: str | None = None

    def __init__(
        self, client: MlflowClient, cache_folder: str, experiment_id: str, run_name: str
    ) -> None:
        self._client = client

        if not FileNameValidator.is_valid(run_name):
            run_name = FileNameValidator.sanitize(run_name)
        self._run_name = run_name

        self._store_path = Path(cache_folder, run_name)
        self._experiment_id = experiment_id

    def get_run(self):
        return self._run

    def get_new_version(self, run_name: str) -> int:
        filter_string = (
            f'attributes.run_name = "{run_name}" AND attributes.status = "FINISHED"'
        )

        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
            order_by=["tags.version DESC"],
        )
        try:
            run = runs[0]
            version = int(run.data.tags["version"]) + 1
            return version

        except (IndexError, KeyError, ValueError):
            return 0

    def check_need_update(self, digest: str) -> bool:
        filter_string = (
            # f'attributes.run_name = "{self._run_name}" AND '
            f'dataset.digest = "{digest}" AND attributes.status = "FINISHED"'
        )
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
        )
        try:
            _ = runs[0]
            return False
        except IndexError:
            return True

    def _get_or_create_run(self) -> Run:
        if not self._run:
            tags: dict[str, Any] = {}
            tags[MLFLOW_RUN_NAME] = self._run_name

            version = self.get_new_version(self._run_name)
            tags["version"] = str(version)

            resolved_tags = context_registry.resolve_tags(tags)
            self._run = self._client.create_run(
                experiment_id=self._experiment_id, tags=resolved_tags
            )
            self._client.set_terminated(self._run.info.run_id, status="RUNNING")

        return self._run

    def _pd_params(self, elem: PdDataset) -> dict[str, int | float]:
        """
        A profile of the dataset. May be ``None`` if a profile cannot be computed.
        """
        return {
            "num_rows": elem.count_rows,
            "num_elements": elem.count_cols,
        }

    @override
    def VisitPdDataset(self, elem: PdDataset):
        # Подразумевается, что датасет уже был перемещён в папку кэша

        if not self.check_need_update(elem.digest):
            return

        run = self._get_or_create_run()

        artifact_uri = run.info.artifact_uri
        if not isinstance(artifact_uri, str):
            raise RuntimeError(f"Bad artifact_uri {artifact_uri}")

        self._client.log_artifact(
            run_id=run.info.run_id,
            local_path=elem.file_path,
            artifact_path=self._root_artifact_path,
        )

        raw_source = os.path.join(
            artifact_uri,
            str(self._root_artifact_path),
            os.path.basename(elem.file_path),
        )

        source: DatasetSource = resolve_dataset_source(raw_source)
        flow_pd_dataset = PandasDataset(
            df=elem.getDataset(),
            source=source,
            name=elem.name,
            digest=elem.digest,
            targets=elem.targets,
            predictions=elem.predictions,
        )

        tags_to_log = []
        if elem.context != DsContext.EMPTY:
            tags_to_log.append(
                InputTag(key=MLFLOW_DATASET_CONTEXT, value=str(elem.context.value))
            )

        dataset_input = DatasetInput(
            dataset=flow_pd_dataset._to_mlflow_entity(),  ## pyright: ignore[reportPrivateUsage]
            tags=tags_to_log,
        )

        self._client.log_inputs(run.info.run_id, [dataset_input])

        self._results.update({elem.name: raw_source})

    @override
    def VisitGroupDataset(self, elem: GroupDataset):
        self._root_artifact_path = "datasets"
        for dataset in elem.datasets:
            dataset.Accept(visitor=self)
        self._root_artifact_path = None

    def get_results(self):
        return self._results
