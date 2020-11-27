from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from LieAlgebraElement import LieAlgebraElement

class LieAlgebra(ABC):
    @abstractmethod
    def identity(self) -> 'LieAlgebraElement':
        pass

    @abstractmethod
    def belongs_to_algebra(self, element: np.ndarray) -> bool:
        pass

    @abstractmethod
    def adjoint(self, algebra_element1, algebra_element2) -> 'LieAlgebraElement':
        pass

    @abstractmethod
    def bracket(self, algebra_element1, algebra_element2) -> 'LieAlgebraElement':
        return self.adjoint(algebra_element1, algebra_element2)
