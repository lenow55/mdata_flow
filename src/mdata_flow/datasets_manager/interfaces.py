from abc import ABC, abstractmethod


# Interface DatasetVisitor
class DatasetVisitor(ABC):
    @abstractmethod
    def Visit(self, elem: "IDataset") -> None:
        pass


# Interface
class IDataset(ABC):
    @abstractmethod
    def Accept(self, visitor: DatasetVisitor) -> None:
        pass
