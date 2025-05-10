from abc import ABC
from pathlib import Path
from typing import final

import pandas as pd
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema  # pyright: ignore[reportPrivateUsage]
from typing_extensions import override

from mdata_flow.datasets_manager.context import DsContext
from mdata_flow.datasets_manager.interfaces import DatasetVisitor, IDataset


# Abstract Dataset class
class Dataset(IDataset, ABC):
    """
    Это класс датасета

    Parameters
    ----------
    name (`str`, *required*)
        Имя датасета, используется для работы в группе датасетов
        должно быть уникально в группе датасетов
        в конце логируется в имя log_input

    schema: (`mlflow.types.schema.Schema`, *required*)
        Схема датасета полученная из mlflow

    targets: (`str`, *optional*, defaults to None)
        Имя колонки с целевой меткой, может быть не указано

    predictions: (`str`, *optional*, defaults to 0)
        Имя колонки с прогнозом целевой метки, может быть не указано

    context: (`mdata_flow.datasets_manager.context.DsContext`, *optional*, defaults to DsContext.EMPTY)
        Контекст использования датасета: Обучение, тест, валидация, пустой

    Returns
    -------

    """

    # путь до файла в кэше
    _file_path: Path | None = None

    # Хэш сумма датасета
    _digest: str | None = None

    # Схема датасета извлечённая при помощи mlflow
    schema: Schema
    targets: str | None
    predictions: str | None
    context: DsContext

    def __init__(
        self,
        name: str,
        schema: Schema,
        targets: str | None = None,
        predictions: str | None = None,
        context: DsContext = DsContext.EMPTY,
    ):
        super().__init__()
        self.name: str = name
        self.schema = schema
        self.targets = targets
        self.predictions = predictions
        self.context = context

    @property
    def digest(self):
        """The digest property"""
        if not self._digest:
            raise RuntimeError("Compute digest before")
        return self._digest

    @digest.setter
    def digest(self, value: str):
        self._digest = value

    @property
    def file_path(self):
        """File path"""
        if not self._file_path:
            raise RuntimeError("Save file before")
        return self._file_path

    @file_path.setter
    def file_path(self, value: Path):
        self._file_path = value


# Concrete Dataset classes
@final
class GroupDataset(IDataset):
    def __init__(self, name: str, datasets: list[IDataset]):
        super().__init__()
        self.name = name
        self.datasets: list[IDataset] = datasets

    @override
    def Accept(self, visitor: DatasetVisitor) -> None:
        visitor.Visit(self)


@final
class PdDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
        context: DsContext = DsContext.EMPTY,
    ):
        self._dataset: pd.DataFrame = dataset
        super().__init__(
            name=name,
            schema=_infer_schema(dataset),
            targets=targets,
            predictions=predictions,
            context=context,
        )

    @override
    def Accept(self, visitor: DatasetVisitor) -> None:
        visitor.Visit(self)

    def getDataset(self) -> pd.DataFrame:
        return self._dataset
