import pandas as pd
from evidently.pipeline.column_mapping import ColumnMapping

from mdata_flow.config import DatasetStoreSettings
from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset
from mdata_flow.datasets_manager.context import DsContext
from mdata_flow.datasets_manager.interfaces import IDataset
from mdata_flow.datasets_manager.manager import DatasetManager
from mdata_flow.datasets_manager.visitors import (
    CSVSaverDatasetVisitor,
    PreviewUploaderVisitor,
)
from mdata_flow.evidently_ext import (
    CountByCategoryReportVisitor,
    DataQualityReportVisitor,
)
from mdata_flow.plotly_ext import (
    PlotlyBoxplotVisitor,
    PlotlyCorrVisitor,
    PlotlyDensityVisitor,
)


def main():
    ds_config = DatasetStoreSettings()  ## pyright: ignore[reportCallIssue]

    print(ds_config.model_dump_json(indent=2))
    full_df = pd.read_csv("eval_data.csv")

    train = full_df.sample(frac=0.8, random_state=47)
    # немного изменим обучающий сет
    test = full_df.drop(
        train.index  ##pyright: ignore[reportCallIssue,reportArgumentType]
    )

    train = train[:-10]

    if not isinstance(test, pd.DataFrame):
        raise RuntimeError("Bad test DataFrame")

    if not isinstance(train, pd.DataFrame):
        raise RuntimeError("Bad train DataFrame")

    datasets: list[IDataset] = [
        PdDataset("df_train", train, targets="label", context=DsContext.TRAIN),
        PdDataset("df_test", test, targets="label", context=DsContext.TEST),
    ]
    composite = GroupDataset(name="test-group", datasets=datasets)
    saver_v = CSVSaverDatasetVisitor(compression="zstd")

    manager = DatasetManager(ds_config, saver_v)
    manager.setup()
    result = manager.register_datasets(composite, run_name="test-upload")

    if result:
        print(manager.get_results())
    else:
        print("датасеты не нуждаются в обновлении")
        exit(0)

    manager.register_extra_uploaders(
        [
            PlotlyCorrVisitor(),
            PlotlyBoxplotVisitor(x_col="label", y_col="Age"),
            PlotlyBoxplotVisitor(x_col="label", y_col="Capital Loss"),
            PlotlyBoxplotVisitor(x_col="Sex", y_col="Capital Loss"),
            PlotlyDensityVisitor(
                categorical_col="Sex",
                numeric_col="Hours per week",
                labels_map={0: "Male", 1: "Female"},
            ),
            PlotlyDensityVisitor(
                categorical_col="Marital Status",
                numeric_col="Hours per week",
            ),
        ]
    )

    preview = PreviewUploaderVisitor(count=25)
    manager.register_extra_uploaders([preview])

    maping = ColumnMapping(
        target="label",
        prediction=None,
        numerical_features=[
            "Age",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
        ],
        categorical_features=[
            "Workclass",
            "Education-Num",
            "Sex",
            "Race",
            "Country",
            "Marital Status",
            "Occupation",
            "Relationship",
        ],
    )
    manager.register_extra_uploaders(
        [
            DataQualityReportVisitor(column_maping=maping),
            CountByCategoryReportVisitor(column_maping=maping),
        ]
    )

    manager.finish_upload()


if __name__ == "__main__":
    main()
