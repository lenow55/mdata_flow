from abc import ABC, abstractmethod
from typing_extensions import override
from mdata_flow.datasets_manager.interfaces import DatasetVisitor, IDataset
from mdata_flow.datasets_manager.composites import GroupDataset, PdDataset


# Base Abstract Typed DatasetVisitor
class TypedDatasetVisitor(DatasetVisitor, ABC):
    @abstractmethod
    def VisitPdDataset(self, elem: PdDataset):
        pass

    @abstractmethod
    def VisitGroupDataset(self, elem: GroupDataset):
        pass

    @override
    def Visit(self, elem: IDataset) -> None:
        if isinstance(elem, PdDataset):
            self.VisitPdDataset(elem=elem)
        elif isinstance(elem, GroupDataset):
            self.VisitGroupDataset(elem=elem)
        else:
            raise RuntimeError(f"Not known dataset type: {type(elem)}")
