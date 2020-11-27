"""
"""
from LieGroup import LieGroup
from LieGroupElement import LieGroupElement
from LieAlgebra import LieAlgebra
from LieAlgebraElement import LieAlgebraElement
import numpy as np
import scipy
import copy

class so3element(LieAlgebraElement):
    representation: np.ndarray
    def __init__(self, _vec: np.ndarray):
        """
        Constructs a 3x3 skew symmetric matrix from a 3d vector _vec.
        """
        self.representation = np.zeros((3, 3))
        self.representation[1, 2] = -_vec[0]
        self.representation[0, 2] = +_vec[1]
        self.representation[0, 1] = -_vec[2]
        self.representation -= np.transpose(self.representation)

    def copy(self) -> 'so3element':
        g = so3element(np.zeros(3))
        g.representation = copy.deepcopy(self.representation)
        return g

class SO3element(LieGroupElement):
    """
    An element of the SO(3) group, with a representation of a 3x3 matrix.
    """
    representation: np.ndarray
    group: LieGroup
    def __init__(self, _config: np.ndarray = np.eye(3)):
        self.group = SO3()
        if self.group.belongs_to_group(_config):
            self.representation = _config
        else:
            raise ValueError("no SO3 element")

    @property
    def orientation(self) -> np.ndarray:
        """
        """
        return self.representation[:3, :3]

    @orientation.setter
    def orientation(self, _or: np.ndarray) -> None:
        """
        """
        if not self.group.belongs_to_group(_or):
            raise ValueError("orientation arg does not have the correct dimensions")
        self.representation = _or

    def __mul__(self, other: 'SO3element') -> 'SO3element':
        """
        """
        return SO3element(np.dot(self.representation, other.representation))

    def __rmul__(self, other: 'SO3element') -> 'SO3element':
        """
        """
        return SO3element(np.dot(other.representation, self.representation))

    def __getitem__(self, key: int) -> np.ndarray:
        """
        Returns one of the orthonormal vectors that define the orientation.
        """
        return self.representation[:,key]

    def inv(self) -> 'SO3element':
        """
        """
        return SO3element(np.transpose(self.representation))

    def __str__(self) -> None:
        """
        """
        return self.representation.__str__()

    def __eq__(self, other: 'SO3element') -> bool:
        """
        """
        return np.allclose(self.representation, other.representation)

    def __ne__(self, other: 'SO3element') -> bool:
        """
        """
        return not np.allclose(self.representation, other.representation)

    def copy(self) -> 'SO3element':
        return copy.deepcopy(self)

class so3(LieAlgebra):
    """
    Lie algebra of the special orthogonal group, containing all 3x3 skew-symmetric matrices.
    """
    def __init__(self):
        pass

    @classmethod
    def identity(cls) -> so3element:
        return so3element(np.zeros(3))

    @classmethod
    def belongs_to_algebra(self, mat: np.ndarray) -> bool:
        return np.allclose(mat, -np.transpose(mat))

    @classmethod
    def adjoint(cls, element1: so3element, element2: so3element) -> so3element:
        return element1*element2 - element2*element1

    @classmethod
    def uniform_random(cls) -> so3element:
        """
        Makes this object into a uniformly random generated rotation axis, by making a spherically
        symmetric pdf, then normalising a random vector drawn from that, then multiplying that
        vector with a uniformly drawn random angle, and turning that into a matrix.
        """
        v = np.random.multivariate_normal(np.zeros(3), np.eye(3))
        normv = np.linalg.norm(v)
        v /= normv
        v *= np.random.uniform(2.*np.pi)

        return so3element(v)

    @classmethod
    def bracket(cls, g: so3element, h: so3element) -> so3element:
        return cls.adjoint(g, h)

    @classmethod
    def vec(cls, g: so3element) -> np.ndarray:
        """
        Makes a 3D vector out of an so3 element
        """
        return np.array([-g.representation[1, 2], g.representation[0, 2], -g.representation[0, 1]])

    @classmethod
    def hat(cls, w: np.ndarray) -> so3element:
        """
        Makes an so3 element out of a 3D vector
        """
        return so3element(w)

class SO3(LieGroup):
    """
    The special orthogonal group of dimension 3.
    """
    algebra: LieAlgebra = so3
    #elements:
    def __init__(self):
        self.algebra = so3()

    @classmethod
    def identity(cls) -> SO3element:
        """
        Returns the SO3 group identity.
        """
        return SO3element(np.eye(3))

    @classmethod
    def belongs_to_group(self, mat: np.ndarray) -> bool:
        return np.isclose(np.linalg.det(mat), 1.0)\
                and np.allclose(np.dot(mat, np.transpose(mat)), np.eye(3), atol=1e-9)

    @classmethod
    def multiplication(cls, element1: SO3element, element2: SO3element) -> SO3element:
        return element1*element2

    @classmethod
    def inverse(cls, element: SO3element) -> SO3element:
        return element.inv()

    @classmethod
    def uniform_random(cls) -> SO3element:
        """
        Makes a uniformly random generated rotation matrix.
        """
        return cls.exp(cls.algebra.uniform_random().copy())

    @classmethod
    def exp(cls, e: so3element) -> SO3element:
        """
        The exponential map of the group that takes an element of the Lie algebra and makes it into
        an element of the Lie group SO3. This is done with the Rodrigues formula.
        """
        axis: np.ndarray = cls.algebra.vec(e)
        angle: float = np.linalg.norm(axis)
        axis /= angle
        #return SO3(np.cos(angle)*np.eye(3) + np.sin(angle)/angle*g.mat
        #+ (1.-np.cos(angle))/angle**2 * np.outer(axis, axis))
        return SO3element(np.cos(angle)*np.eye(3)\
                + cls.algebra.hat(np.sin(angle)*axis).representation\
                + (1.-np.cos(angle)) * np.outer(axis, axis))

    @classmethod
    def log(cls, G: SO3element) -> so3element:
        """
        The logarithmic map of the group that takes an element of SO3 and gives an element of the
        Lie algebra so3.
        """
        try:
            angle: float = np.arccos(0.5*(np.trace(G.representation)-1.0))
            #print(angle)

            if angle == np.nan or np.isclose(angle, 0.0):
                raise ArithmeticError

            g = so3element(np.zeros(3))

            #print(G.mat-np.transpose(G.mat))
            g.representation = (angle/(2.*np.sin(angle))\
                    *(G.representation-np.transpose(G.representation)))
            return g
        except ArithmeticError:
            print('angle zero')

    @classmethod
    def sqrt(cls, G: SO3element) -> SO3element:
        """
        """
        #print('G', G)
        #print('logG', log(G))
        #print('veclogG', vec(log(G)))
        #print('0.5*veclogG', .5*vec(log(G)))
        #print('hat0.5*veclogG', hat(.5*vec(log(G))))
        return cls.exp(cls.algebra.hat(0.5*cls.algebra.vec(cls.log(G))))

    @classmethod
    def jacobian_det(cls, val: so3element) -> float:
        """
        Calculates the jacobian from an so3 element. In our specific case of Curves+ coordinates
        in radians, it's abs(sinc^2(angle/2)).
        """
        angle: float = np.norm(cls.algebra.vec(val))
        return np.abs(2.*(1.-np.cos(angle))/np.power(angle, 2.))

    @classmethod
    def conjugate(cls, G: SO3element, H: SO3element) -> SO3element:
        """
        Conjugate G by H.
        """
        return H*G*H.inv()

    @classmethod
    def Adjoint(cls, Gsub: SO3element, g: so3element) -> so3element:
        """
        Adjoint action of the Lie group on the Lie algebra.
        """
        return so3element(np.dot(Gsub.representation, g.representation))




if __name__ == "__main__":
    v = np.array([np.sqrt(3.), np.sqrt(2.), 15.])
    print(v)
    e = so3element(v)
    rot = SO3.exp(e) #is correct
    print(rot)
    #print(vec(log(G)))
