from abc import ABC, abstractmethod


# Interface DatasetVisitor
class DatasetVisitor(ABC):
    @abstractmethod
    def Visit(self, elem: "IDataset") -> None:
        pass


# Interface
class IDataset(ABC):
    """
    name (`str`, *required*)
        Имя датасета или группы, используется для построения путей
        должно быть уникально в группе датасетов
        в конце логируется в имя log_input
    """

    name: str

    @abstractmethod
    def Accept(self, visitor: DatasetVisitor) -> None:
        pass
