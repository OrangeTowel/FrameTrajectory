"""
"""
from LieGroup import LieGroup
from LieGroupElement import LieGroupElement
from LieAlgebra import LieAlgebra
from LieAlgebraElement import LieAlgebraElement

from typing import List
import numpy as np
import scipy.linalg as sl
from SO3 import SO3, so3, SO3element, SO3element
import copy

class se3element(LieAlgebraElement):
    def __init__(self, _vec: np.ndarray):
        pass

class SE3element(LieGroupElement):
    representation: np.ndarray
    group: LieGroup
    def __init__(self, _or: SO3element = SO3.identity(), _pos: np.ndarray = np.zeros(3)):
        self.group = SE3()
        self.representation = np.eye(4)
        self.orientation = _or
        self.position = _pos

    @property
    def orientation(self) -> SO3element:
        return SO3element(self.representation[:3, :3].copy())

    @orientation.setter
    def orientation(self, _or: SO3element) -> None:
        self.representation[:3, :3] = _or.representation

    @property
    def position(self) -> np.ndarray:
        return self.representation[:3, 3].copy()

    @position.setter
    def position(self, _pos: np.ndarray) -> None:
        self.representation[:3, 3] = _pos

    def __mul__(self, other: 'SE3element') -> 'SE3element':
        g = SE3element()
        g.orientation = self.orientation*other.orientation
        g.position = np.dot(self.orientation.representation, other.position) + self.position
        return g

    def __rmul__(self, other: 'SE3element') -> 'SE3element':
        g = SE3element()
        g.orientation = other.orientation*self.orientation
        g.position = np.dot(other.orientation.representation, self.position) + other.position
        return g

    def __str__(self) -> str:
        return self.representation.__str__()

    def __eq__(self, other: 'SE3element') -> bool:
        return self.orientation == other.orientation and np.allclose(self.position, other.position)

    def __ne__(self, other: 'SE3element') -> bool:
        return not (self.orientation == other.orientation and np.allclose(self.position, other.position))

    def inv(self) -> 'SE3element':
        g = SE3element()
        g.orientation = self.orientation.copy().inv()
        g.position = -np.dot(self.orientation.copy().inv().representation, self.position)
        return g

    def __str__(self) -> str:
        """
        """
        return self.representation.__str__()

    def __eq__(self, other: 'SE3element') -> bool:
        """
        """
        return np.allclose(self.representation, other.representation)

    def __ne__(self, other: 'SE3element') -> bool:
        """
        """
        return not np.allclose(self.representation, other.representation)

    def copy(self) -> 'SE3element':
        g = SE3element()
        g.representation = copy.deepcopy(self.representation)
        #return g
        return copy.deepcopy(self)

class se3(LieAlgebra):
    def __init__(self):
        pass

    @classmethod
    def identity(self) -> se3element:
        return se3element(np.zeros((4,4)))

    @classmethod
    def belongs_to_algebra(self, mat: np.ndarray) -> bool:
        return np.isclose(mat[3,3], 0.0) and np.allclose(mat[:3, 3], np.zeros(3)) and np.allclose(mat[:3,:3], -np.transpose(mat[:3,:3]))

    @classmethod
    def adjoint(self, element1: se3element, element2: se3element) -> se3element:
        return element1*element2 - element2*element1

    @classmethod
    def bracket(self, g: se3element, h: se3element) -> se3element:
        return self.adjoint(g, h)

class SE3(LieGroup):
    """
    Implements the Special Euclidean group in 3 dimensions
    """
    algebra: LieAlgebra = se3
    def __init__(self):
        self.algebra = se3()
        pass

    @classmethod
    def identity(self) -> 'SE3element':
        return SE3element()

    @classmethod
    def fromOrAndPos(self, _orientation: SO3element = SO3element(), _position: np.ndarray = np.zeros(3)) -> SE3element:
        g = SE3element()
        g.orientation = _orientation
        g.position = _position
        return g

    @classmethod
    def belongs_to_group(self, mat: np.ndarray) -> bool:
        return np.isclose(mat[3,3], 1.0) and np.allclose(mat[3,:3], np.zeros(3)) and np.isclose(np.linalg.det(mat[:3,:3]), 1.0) and np.allclose(np.dot(mat[:3,:3], np.transpose(mat[:3,:3])), np.eye(3))

    @classmethod
    def multiplication(self, element1: SE3element, element2: SE3element) -> SE3element:
        return element1*element2

    @classmethod
    def inverse(self, element: SE3element) -> SE3element:
        return element.inv()

    #def sqrt(self) -> 'SE3':
        #return SE3(scipy.linalg.sqrtm(self.mat))

    @classmethod
    def log(self, g: SE3element) -> se3element:
        m = np.real(sl.logm(g.representation))
        h = self.algebra.identity()
        h.representation = m
        return h

    @classmethod
    def exp(self, g: se3element) -> SE3element:
        expo = sl.expm(g.representation)
        return SE3element(SO3element(expo[:3,:3]), expo[:3,3])

    @classmethod
    def conjugate(self, G1: SE3element, G2: SE3element) -> SE3element:
        """
        The ADjoint action of the group on itself
        """
        return G1*G2*G1.inv()

    @classmethod
    def Adjoint(self, G: SE3element, g: se3element) -> se3element:
        return np.dot(np.dot(G.representation, g.representation), G.inv().representation)

    @classmethod
    def midframe(self, g1: SE3element, g2: SE3element) -> SE3element:
        """
        Calculates the midframe of two SE3 elements.
        """
        g1R = g1.orientation
        g2R = g2.orientation
        midR = g1R*SO3.sqrt(g1R.inv()*g2R)
        return SE3element(midR, 0.5*(g1.position+g2.position))

    @classmethod
    def degrees_of_freedom(self, g1: SE3element, g2: SE3element) -> List[float]:
        """
        Calculates the six degrees of freedom between two SE3 elements.
        """
        g1R = g1.orientation
        g2R = g2.orientation
        rot_dof = list(so3.vec(SO3.sqrt(g1R.inv()*g2R)))
        midR = g1R*SO3.sqrt(g1R.inv()*g2R)
        trans_dof = np.dot(midR.inv().representation, g2.position-g1.position)
        rot_dof.extend(trans_dof)
        #print('r', rot_dof)
        #print('t', trans_dof)
        return rot_dof

    @classmethod
    def degrees_of_freedomSE3(self, g1: SE3element, g2: SE3element) -> SE3element:
        """
        Calculates the six degrees of freedom between two SE3 elements.
        """
        g1R = g1.orientation
        g2R = g2.orientation
        rot_dof = g1R.inv()*g2R
        midR = g1R*SO3.sqrt(g1R.inv()*g2R)
        trans_dof = np.dot(midR.inv().representation, g2.position-g1.position)
        return SE3element(rot_dof, trans_dof)


if __name__ == "__main__":
    E = SE3.identity()
    e = se3.identity()
