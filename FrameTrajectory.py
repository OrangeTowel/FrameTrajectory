import sys
import os
from typing import Dict, List, Any, Union
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

#sys.path.insert(1, os.path.join(sys.path[0], '/home/debruin/PhD/Code/DNA_python_scripts/'))

from SE3 import SE3element, SE3
from SO3 import SO3element, SO3, so3element, so3

def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()

class FrameTrajectory:
    """
    Class that implements the trajectory of a single coordinate frame.
    """
    name: str
    n_timeframes: int
    dt: float
    time: np.ndarray
    frames: np.ndarray
    attributes: Dict[str, Any] #to put any other information in

    def __init__(self, name="", n_timeframes: int = 0, dt: float = 1e-15):
        self.name = name
        self.n_timeframes = n_timeframes
        self.dt = dt
        self.time = np.array([dt*i for i in range(next_power_of_2(n_timeframes))])
        self.frames = np.repeat(np.eye(4)[np.newaxis,:,:], next_power_of_2(n_timeframes), axis=0)#[SE3element() for x in range(timeframes)]
        self.attributes = {}

    def append(self, g: SE3element) -> None:
        """
        Appends a frame to the trajectory.
        """
        if self.n_timeframes >= self.frames.shape[0]:
            tmp_frames = np.zeros((next_power_of_2(self.frames.shape[0]+1), 4, 4))
            tmp_frames[0:self.frames.shape[0],:,:] = self.frames
            self.frames = tmp_frames
        self.n_timeframes += 1
        self.frames[self.n_timeframes-1] = g.representation

    def __len__(self) -> int:
        return self.n_timeframes

    def __iter__(self):
        return iter(self.frames)

    def __str__(self) -> str:
        return f"<FrameTrajectory {self.name} with {self.n_timeframes} frames.>"

    def __add__(self, other: 'FrameTrajectory') -> 'FrameTrajectory':
        """
        Constructs a new trajectory which contains the two trajectories together by appending one to the other.
        """
        new_traj = FrameTrajectory(name=self.name+'+'+other.name, n_timeframes=self.n_timeframes+other.n_timeframes)
        new_traj.frames[0:self.n_timeframes] = self.frames[0:self.n_timeframes]
        new_traj.frames[self.n_timeframes : self.n_timeframes+other.n_timeframes] = other.frames[0:other.n_timeframes]

        return new_traj

    def __iadd__(self, other: 'FrameTrajectory') -> 'FrameTrajectory':
        """
        Adds two trajectories together by appending one to the left one.
        """
        if self.n_timeframes+other.n_timeframes >= self.frames.shape[0]:
            tmp_frames = np.zeros((next_power_of_2(self.n_timeframes+other.n_timeframes+1), 4, 4))
            tmp_frames[0:self.n_timeframes] = self.frames[0:self.n_timeframes]
            tmp_frames[self.n_timeframes : self.n_timeframes+other.n_timeframes] = other.frames[0:other.n_timeframes]
            self.n_timeframes += other.n_timeframes
            self.frames = tmp_frames

        return self

    def __mul__(self, other: 'FrameTrajectory') -> 'FrameTrajectory':
        """
        Multiply the elements from one trajectory with the elements from the other.
        The trajectories need to have the same length.
        """
        if len(self) != len(other):
            raise ValueError("Trajectories do not have the same size")

        new_traj = FrameTrajectory(name=self.name+'*'+other.name, n_timeframes=self.n_timeframes)
        new_traj.frames = np.matmul(self.frames[:self.n_timeframes], other.frames[:other.n_timeframes])

        return new_traj

    def __getitem__(self, key: Union[int, slice]) -> Union[SE3element, 'FrameTrajectory']:
        """
        Overloads the [] operator for obtaining values. key is and int or a slice.
        """
        if isinstance(key, int):
            frame = SE3element()
            frame.representation = self.all_frames[key]
            return frame

        elif isinstance(key, slice):
            n_timeframes: int = len(range(*key.indices(len(self.all_frames))))
            new_traj = FrameTrajectory(name=self.name, n_timeframes=n_timeframes)
            new_traj.all_frames[:n_timeframes] = self.all_frames[key]

            return new_traj
        else:
            raise ValueError("incorrect index in FrameTrajectory __getitem__")

    #def __setitem__(self, key: Union[int, slice], value: Union[SE3element, List[SE3element]]) -> None:
    def __setitem__(self, key: Union[int, slice], value: Union[SE3element, 'FrameTrajectory']) -> None:
        """
        Overloads the [] operator for setting values. key is and int or a slice.
        """
        if isinstance(key, int) and isinstance(value, SE3element):
            self.all_frames[key] = value.representation
        elif isinstance(key, slice):
            n_timeframes: int = len(range(*key.indices(len(self.all_frames))))
            if n_timeframes != len(value):
                raise ValueError("different lengths of indices and traj in FrameTrajectory __getitem__")
            #tmp_frames: np.ndarray = np.zeros(n_timeframes,4,4)
            #for f in value:
                #tmp_frames[i] = f.representation
            #self.all_frames[key] = tmp_frames
            self.all_frames[key] = value.all_frames

    @property
    def all_frames(self) -> np.ndarray:
        return self.frames[:self.n_timeframes]

    @all_frames.setter
    def all_frames(self, value) -> None:
        self.frames[:self.n_timeframes] = value

    @property
    def orientations(self) -> np.ndarray:
        """
        Returns an array with all the rotation elements of this trajectory.
        """
        #return self.frames[:self.n_timeframes,:3,:3]
        return self.all_frames[:,:3,:3]

    @orientations.setter
    def orientations(self, value: np.ndarray) -> None:
        """
        Returns an array with all the rotation elements of this trajectory.
        """
        #return self.frames[:self.n_timeframes,:3,:3]
        self.all_frames[:,:3,:3] = value

    @property
    def positions(self) -> np.ndarray:
        """
        Returns an array with all the translation elements of this trajectory.
        """
        #return self.frames[:self.n_timeframes, :3, 3:]
        return self.all_frames[:, :3, 3:]

    @positions.setter
    def positions(self, value: np.ndarray) -> None:
        """
        Returns an array with all the translation elements of this trajectory.
        """
        #return self.frames[:self.n_timeframes, :3, 3:]
        self.all_frames[:, :3, 3:] = value

    def inv(self) -> 'FrameTrajectory':
        """
        Compute the inverse of the current trajectory by making a new trajectory with the
        inverses of all frames of the current trajectory.
        """
        inv_traj = FrameTrajectory(name="inv("+self.name+")", n_timeframes=self.n_timeframes)

        inv_traj.orientations = np.transpose(self.orientations, (0, 2, 1))
        inv_traj.positions = -np.matmul(inv_traj.orientations, self.positions)

        return inv_traj

    def left_multiply(self, g: SE3element) -> 'FrameTrajectory':
        """
        Left multiply all elements in this trajectory with the given SE3 element and return
        the new trajectory.
        """
        new_traj = FrameTrajectory(name='g_'+self.name, n_timeframes=self.n_timeframes)
        grep = g.representation
        #tmp_frames = np.matmul(grep, self.frames[:self.n_timeframes])
        #new_traj.frames[:self.n_timeframes] = tmp_frames
        tmp_frames = np.matmul(grep, self.all_frames)
        new_traj.all_frames = tmp_frames

        return new_traj

    def right_multiply(self, g: SE3element) -> 'FrameTrajectory':
        """
        Right multiply all elements in this trajectory with the given SE3 element and
        return the new trajectory.
        """
        new_traj = FrameTrajectory(name=self.name+'_g', n_timeframes=self.n_timeframes)
        #new_traj = FrameTrajectory(name=self.name+f'*{g=}'.split('=')[0], timeframes=self.n_timeframes) #f-string debugging to get name of variable
        grep = g.representation
        #tmp_frames = np.matmul(self.frames[:self.n_timeframes], grep)
        #new_traj.frames[:self.n_timeframes] = tmp_frames
        tmp_frames = np.matmul(self.all_frames, grep)
        new_traj.all_frames = tmp_frames

        return new_traj

    def fromJSON(self, json_trajectory: Dict[str, Any]) -> 'FrameTrajectory':
        self.name = json_trajectory["name"]
        self.n_timeframes = 0
        for key, value in json_trajectory.items():
            #if key is not "name" and key is not "n_timeframes" and key is not "frames":
            if key not in vars(self):
                self.attributes[key] = value

        for f in json_trajectory["frames"]:
            frame: SE3element = SE3().fromOrAndPos(SO3element(np.array(f["R"])), np.array(f["r"]))#.representation
            self.append(frame)

        return self

    def toJSON(self) -> Dict[str, Any]:
        """
        Outputs the data in this trajectory in JSON format.
        """
        json: Dict[str, Any] = {}
        json["name"] = self.name
        json["n_timeframes"] = self.n_timeframes
        json["frames"] = []
        for key, value in self.attributes.items():
            json[key] = value

        positions: np.ndarray
        if self.n_timeframes == 1:
            positions = np.reshape(self.positions, (1,3))
        else:
            positions = np.squeeze(self.positions)
        orientations = self.orientations
        for index, frame in enumerate(self.all_frames):
            json["frames"].append({
                "R": orientations[index].tolist(), #TODO is this correct???
                "r": positions[index].tolist(),
                "timeframe": index
                })

        return json

    def relative_to(self, other: "FrameTrajectory") -> 'FrameTrajectory':
        """
        Calculates what the trajectory relative to another trajectory is, with left multiplication
        """
        if len(self) != len(other):
            raise ValueError("Trajectories do not have the same size")

        new_traj: FrameTrajectory = other.inv()*self
        new_traj.name = self.name+'_rel_to_'+other.name

        return new_traj

    def mean(self) -> SE3element:
        """
        Returns the mean of this trajectory.
        """
        #can do two things:
        #   1) take the exponential of the arithmatic mean of the logs of each element
        #   2) calculate the Frechet mean of all the elements (actually the same as 1?
        mean_position = np.mean(self.positions)
        mean_orientation = expSO3(np.mean(logSO3(self.orientations)))
        #mean_orientation = SO3.exp(np.mean(np.array([SO3.vec(SO3.log(frame.orientation)) for frame in self.frames])))

        return SE3element(mean_orientation, mean_position)

    def jacobian_determinants(self) -> np.ndarray:
        """
        Return the jacobian determinant for each timeframe in this trajectory.
        This is necessary for the computation of the correct means and covariances.
        """
        angs = angles(logSO3(self.orientations))

        return np.abs(2.*np.divide(np.ones(angs.size[0])-np.cos(angs),np.square(angs)))

def logSO3(orientations: np.ndarray) -> np.ndarray:
    """
    Takes an array of n SO3 elements, a numpy array of shape (n, 3, 3),
    returns a (n, 3, 3 ) array of so3 elements.
    """
    #do test whether none have angle of zero.
    angles: np.ndarray
    try:
        angles = np.arccos(0.5*(np.trace(orientations, axis1=1, axis2=2) - 1.))
    except:
        print("Angle is zero in SO3 log.Exiting")
        exit(1)

    #remove the nans (they are zeroes)??!
    angles = np.nan_to_num(angles)

    fracs: np.ndarray
    try:
        fracs = 0.5*np.divide(angles, np.sin(angles))
    except:
        print("Angle is zero in SO3 log.Exiting")
        exit(1)

    return np.multiply(fracs[:, np.newaxis, np.newaxis], orientations-np.transpose(orientations, (0, 2, 1)))

def logSE3(traj: FrameTrajectory) -> np.ndarray:
    """
    Takes an array of n SE3 elements, a numpy array of shape (n, 4, 4),
    returns a (n, 4, 4 ) array of se3 elements.
    """
    logsSO3: np.ndarray = logSO3(traj.orientations)
    angs: np.ndarray = angles(logsSO3)
    n: int = traj.n_timeframes
    I: np.ndarray = np.repeat(np.eye(3)[np.newaxis, :, :], n, axis=0)
    inv_angle2: np.ndarray = np.square(np.reciprocal(angs))
    factor: np.ndarray = np.ones(n) - np.divide(np.multiply(angs, np.sin(angs)), 2.*(np.ones(n) - np.cos(angs)))

    inv_V = I - 0.5*logsSO3\
            + np.multiply(inv_angle2[:, np.newaxis, np.newaxis], np.multiply(factor[:, np.newaxis, np.newaxis], np.matmul(logsSO3,logsSO3)))

    O: np.ndarray = np.repeat(np.zeros((4,4))[np.newaxis, :, :], n, axis=0)
    O[:, :3,:3] = logsSO3
    O[:, :3, 3:] = np.matmul(inv_V, traj.positions)

    return O

def so3_to_vec(orientations: np.ndarray) -> np.ndarray:
    """
    Converts a Numpy array of n so3 elements of shape (n, 3, 3) to
    a numpy array of 3-vectors, of shape (n, 3, 1).
    """
    #tmp = np.transpose(orientations, (1,2,0))
    return np.transpose(np.array([-orientations[:,1,2], orientations[:,0,2], -orientations[:,0,1]]))

def angles(asym_matrices: np.ndarray) -> np.ndarray:
    """
    Takes an array of n so3 elements, a numpy array of shape (n, 3, 3),
    returns an n-dim array containing the angles.
    """
    return np.linalg.norm(so3_to_vec(asym_matrices), axis=1)

def expSO3(orientations: np.ndarray) -> np.ndarray:
    """
    Takes an array of n so3 elements, a numpy array of shape (n, 3, 3),
    returns a (n, 3, 3 ) array of SO3 elements.
    """
    vecs: np.ndarray = so3_to_vec(orientations)
    angs: np.ndarray = angles(orientations)
    n: int = orientations.shape[0]
    I: np.ndarray = np.repeat(np.eye(3)[np.newaxis,:,:], n, axis=0)

    cos: np.ndarray = np.cos(angs)
    inv_angle: np.ndarray = np.reciprocal(angs)

    outer_vecs = vecs[:,:, np.newaxis]*vecs[:,np.newaxis,:]

    return np.multiply(cos[:, np.newaxis, np.newaxis], I)\
            + np.multiply(np.multiply(inv_angle, np.sin(angs))[:, np.newaxis, np.newaxis], orientations)\
            + np.multiply(np.multiply(np.square(inv_angle), np.ones(n)-cos)[:, np.newaxis, np.newaxis], outer_vecs)

def sqrt(orientations: np.ndarray) -> np.ndarray:
    """
    Takes an array of n SO3 elements, a numpy array of shape (n, 3, 3),
    takes the square root, and returns
    """
    return expSO3(0.5*logSO3(orientations))

def midframes(traj1: FrameTrajectory, traj2: FrameTrajectory) -> FrameTrajectory:
    """
    Calculates the trajectory of the midframe of the two given trajectories.
    The trajectories need to have the same length.
    """
    mf_traj = FrameTrajectory(name='mid('+traj1.name+','+traj2.name+")", n_timeframes=traj1.n_timeframes)

    rel_traj = traj1.inv()*traj2
    sqrt_rel_or = sqrt(rel_traj.orientations)

    mf_orientations: np.ndarray = np.matmul(traj1.orientations, sqrt_rel_or)
    mf_positions: np.ndarray = 0.5*(traj1.positions + traj2.positions)

    mf_traj.orientations = mf_orientations
    mf_traj.positions = mf_positions

    return mf_traj

def degrees_of_freedom(traj1: FrameTrajectory, traj2: FrameTrajectory) -> np.array:
    """
    Calculates the degrees of freedom for the frames of the two given trajectories
    of n timeframes, and returns a Numpy array of shape (n, 6) containing the 6-vectors.
    """
    dof_or_vec = so3_to_vec(logSO3((traj1.inv()*traj2).orientations))
    mf_or = midframes(traj1, traj2).orientations
    rel_pos = (traj2.positions-traj1.positions)
    dof_pos = np.matmul(np.transpose(mf_or, (0, 2, 1)), rel_pos)
    dof_pos = np.reshape(dof_pos, (-1,3))

    return np.concatenate((dof_or_vec, dof_pos), axis=1)

def noncentered_degrees_of_freedom(traj1: FrameTrajectory, traj2: FrameTrajectory) -> np.array:
    """
    Calculates the degrees of freedom for the frames of the two given trajectories
    of n timeframes, and returns a Numpy array of shape (n, 6) containing the 6-vectors.
    The difference with the normal (centered) degrees of freedom is the frame in which
    the translations are expressed. Here, it's the frame of traj2, as opposed to the
    midframe of the two trajectories.
    """
    dof_or_vec = so3_to_vec(logSO3((traj1.inv()*traj2).orientations))
    rel_pos = (traj2.positions-traj1.positions)
    dof_pos = np.matmul(np.transpose(traj2.orientations, (0, 2, 1)), rel_pos)
    dof_pos = np.reshape(dof_pos, (-1,3))

    return np.concatenate((dof_or_vec, dof_pos), axis=1)

#def corrected_degrees_of_freedom(trajectory1, trajectory2) -> List[np.array]:
#    """
#    Calculates the degrees of freedom for the frames of the two given trajectories,
#    and returns a list containing the 6-vectors.
#    """
#    dofs = []
#    for f, g in zip(trajectory1, trajectory2):
#        dof = np.array(SE3().degrees_of_freedom(f, g))
#        J = SO3.jacobian_det(so3.hat(dof[:3]))
#
#        dofs.append(dof/J)
#
#    return dofs

def reciprocal_jacobian_dets(traj1, traj2) -> np.array:
    """
    Calculates the inverse Jacobian determinant for the frames of the two given trajectories,
    and returns a list containing these.
    """
    angs = angles(logSO3(traj2.relative_to(traj1).orientations))

    return np.reciprocal(np.abs(2.*np.divide(np.ones(angs.size)-np.cos(angs),np.square(angs))))

def plot_histogram(DOFs: np.ndarray, *, filename: str, label: str = "", **kwargs) -> None:
    """
    Plots a histogram of the given trajectory in a file with the given filename.
    The six things plotted are the values of the rotation, and the values of the
    translation.
    """
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    #tilt, roll, twist, shift, slide, rise = DOFs
    fig.text(0.01, 0.5, label, fontsize=30)
    #print(kwargs)
    if "label_color" in kwargs:
        fig.text(0.01, 0.5, label, color=kwargs['label_color'], fontsize=30)

    for i in range(0,6):
        ax[i].hist(DOFs[:,i], bins=50)
    #ax[0].hist(tilt, bins=50)
    #ax[1].hist(roll, bins=50)
    #ax[2].hist(twist, bins=50)
    #ax[3].hist(shift, bins=50)
    #ax[4].hist(slide, bins=50)
    #ax[5].hist(rise, bins=50)

    for i in range(0,6):
        ax[i].tick_params(axis='both', which='major', labelsize=30)

    fig.savefig(filename)
    plt.close()

def plot_histogram_traj(traj: FrameTrajectory, *, filename: str, label: str = "", **kwargs) -> None:
    """
    Plots a histogram of the given trajectory in a file with the given filename.
    The six things plotted are the values of the rotation, and the values of the
    translation.
    """
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    #tilt, roll, twist, shift, slide, rise = DOFs
    fig.text(0.01, 0.5, label, fontsize=30)
    #print(kwargs)
    if "label_color" in kwargs:
        fig.text(0.01, 0.5, label, color=kwargs['label_color'], fontsize=30)

    flat = np.concatenate((so3_to_vec(logSO3(traj.orientations)), np.squeeze(traj.positions)), axis=1)
    for i in range(0,6):
        ax[i].hist(flat[:,i], bins=50)

    for i in range(0,6):
        ax[i].tick_params(axis='both', which='major', labelsize=30)

    fig.savefig(filename)
    plt.close()
