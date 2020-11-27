from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class LieAlgebraElement(ABC):
    representation: np.ndarray

    def __mul__(self, factor: float) -> 'LieAlgebraElement':
        return factor*self.representation
    def __rmul__(self, factor: float) -> 'LieAlgebraElement':
        return factor*self.representation

    def __add__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        return self.representation+other.representation
    def __sub__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        return self.representation-other.representation

    def __eq__(self, other: 'LieAlgebraElement') -> bool:
        return np.allclose(self.representation,other.representation)
    def __ne__(self, other: 'LieAlgebraElement') -> bool:
        return not np.allclose(self.representation,other.representation)
