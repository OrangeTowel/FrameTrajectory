from abc import ABC, abstractmethod
import numpy as np
#from typing import None

class LieGroupElement:
    representation: np.ndarray

    @abstractmethod
    def __mul__(self, other_group_element) -> 'LieGroupElement':
        return LieGroupElement(np.dot(self.representation, other_group_element.representation))
    @abstractmethod
    def __rmul__(self, other_group_element) -> 'LieGroupElement':
        return LieGroupElement(np.dot(other_group_element.representation, self.representation))

    @abstractmethod
    def inv(self) -> None:
        return LieGroupElement(np.inverse(self.representation))

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __ne__(self, other) -> bool:
        pass

