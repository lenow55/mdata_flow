from mlflow.client import MlflowClient
from typing_extensions import Any


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
