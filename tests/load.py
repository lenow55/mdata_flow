import pandas as pd

from mdata_flow.config import DatasetStoreSettings
from mdata_flow.datasets_manager.downloader import DatasetDownloader


def main():
    ds_config = DatasetStoreSettings()  ## pyright: ignore[reportCallIssue]

    print(ds_config.model_dump_json(indent=2))

    manager = DatasetDownloader(ds_config)
    manager.setup()
    result = manager.download(
        run_name="test-upload-3", run_version=0, dataset_name="df_test"
    )

    print(result)


if __name__ == "__main__":
    main()
