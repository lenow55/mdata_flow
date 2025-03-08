from abc import ABC
from typing import final

import pandas as pd
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema  # pyright: ignore[reportPrivateUsage]
from typing_extensions import override

from mdata_flow.datasets_manager.interfaces import DatasetVisitor, IDataset
from mdata_flow.datasets_manager.context import DsContext


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

    count_cols: (`int`, *optional*, defaults to 0)
        Количество колонок в датасете
        может быть рассчитано в дочернем классе

    count_rows: (`int`, *optional*, defaults to 0)
        Количество строк в датасете
        может быть рассчитано в дочернем классе

    Returns
    -------

    """

    # временный файл перед расчётом digest
    _temp_file: str | None = None
    # тип файла и путь до файла в кэше
    _file_type: str | None = None
    _file_path: str | None = None

    # Хэш сумма датасета
    _digest: str | None = None

    # Схема датасета извлечённая при помощи mlflow
    schema: Schema

    def __init__(
        self,
        name: str,
        schema: Schema,
        count_cols: int = 0,
        count_rows: int = 0,
    ):
        super().__init__()
        self.name: str = name
        self._count_cols: int = count_cols
        self._count_rows: int = count_rows
        self.schema = schema

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
    def temp_path(self):
        """Temp file path"""
        if not self._temp_file:
            raise RuntimeError("Save file before")
        return self._temp_file

    @temp_path.setter
    def temp_path(self, value: str):
        self._temp_file = value

    @property
    def file_path(self):
        """File path"""
        if not self._file_path:
            raise RuntimeError("Save file before")
        return self._file_path

    @file_path.setter
    def file_path(self, value: str):
        self._file_path = value

    @property
    def count_cols(self):
        """Count cols for df"""
        return self._count_cols

    @count_cols.setter
    def count_cols(self, value: int):
        self._count_cols = value

    @property
    def count_rows(self):
        """Count cols for df"""
        return self._count_rows

    @count_rows.setter
    def count_rows(self, value: int):
        self._count_rows = value

    @property
    def file_type(self):
        """File path"""
        if not self._file_type:
            raise RuntimeError("Save file before")
        return self._file_type

    @file_type.setter
    def file_type(self, value: str):
        self._file_type = value


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

    @override
    def Accept(self, visitor: DatasetVisitor) -> None:
        visitor.Visit(self)

    def getDataset(self) -> pd.DataFrame:
        return self._dataset
