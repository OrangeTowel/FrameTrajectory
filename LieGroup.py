from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class LieGroup(ABC):
    @abstractmethod
    def identity(self) -> 'LieGroup':
        pass

    @abstractmethod
    def belongs_to_group(self, mat: np.ndarray) -> bool:
        pass

    @abstractmethod
    def exp(self, algebra_element) -> Any:
        pass

    @abstractmethod
    def log(self, group_element) -> Any:
        pass

    @abstractmethod
    def Adjoint(self, group_element, algebra_element) -> Any:
        pass

    @abstractmethod
    def multiplication(self, group_element_left, group_element_right) -> Any:
        return group_element_left*group_element_right

    @abstractmethod
    def inverse(self, group_element) -> Any:
        group_element.inverse()

    #@abstractmethod
    #def metric(self, group_element1, group_element2) -> float:
    #    pass
