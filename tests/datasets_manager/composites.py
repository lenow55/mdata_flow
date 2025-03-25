from typing import final, override

import pandas as pd
from mlflow.types.utils import _infer_schema  # pyright: ignore[reportPrivateUsage]

from mdata_flow.datasets_manager.composites import Dataset, PdDataset
from mdata_flow.datasets_manager.context import DsContext
from mdata_flow.datasets_manager.interfaces import DatasetVisitor


@final
class MockPdDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
        context: DsContext = DsContext.EMPTY,
    ):
        super().__init__(
            name=name,
            schema=_infer_schema(dataset),
            count_cols=dataset.shape[1],
            count_rows=dataset.shape[0],
        )
        self._dataset: pd.DataFrame = dataset
        self.targets: str | None = targets
        self.predictions: str | None = predictions
        self.context: DsContext = context
        self.__class__ = PdDataset  ## pyright: ignore[reportAttributeAccessIssue]

    @override
    def Accept(self, visitor: DatasetVisitor) -> None:
        visitor.Visit(self)
        print(type(visitor))

    def getDataset(self) -> pd.DataFrame:
        return self._dataset


if __name__ == "__main__":
    dataset = MockPdDataset(
        name="test", dataset=pd.DataFrame(data=[1, 2], columns=pd.Index(["1"]))
    )

    print(isinstance(dataset, PdDataset))  # pyright: ignore[reportUnnecessaryIsInstance]
