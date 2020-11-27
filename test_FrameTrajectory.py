import unittest
from FrameTrajectory import FrameTrajectory, logSO3, expSO3, sqrt, logSE3, midframes, degrees_of_freedom#, symmetrise
import sys
import os
import numpy as np
import scipy.linalg as sl
#sys.path.insert(1, os.path.join(sys.path[0], '../../DNA_python_scripts'))
import json

from SE3 import SE3element, SE3
from SO3 import SO3element, SO3


class TestFrameTrajectory(unittest.TestCase):
    def test_init(self):
        ft = FrameTrajectory()
        self.assertTrue(ft.name == "")
        self.assertTrue(ft.n_timeframes == 0)
        self.assertTrue(np.allclose(ft.frames, np.repeat(np.eye(4)[np.newaxis,:,:], 1, axis=0)))

        ft = FrameTrajectory("blargh")
        self.assertTrue(ft.name == "blargh")
        self.assertTrue(ft.n_timeframes == 0)
        self.assertTrue(np.allclose(ft.frames, np.repeat(np.eye(4)[np.newaxis,:,:], 1, axis=0)))

        ft = FrameTrajectory("blargh", 100)
        self.assertTrue(ft.name == "blargh")
        self.assertTrue(ft.n_timeframes == 100)
        self.assertTrue(np.allclose(ft.frames, np.repeat(np.eye(4)[np.newaxis,:,:], 128, axis=0)))

    def test_append(self):
        ft = FrameTrajectory("blargh", 4)
        ft.append(SE3element())
        self.assertTrue(ft.name == "blargh")
        self.assertTrue(ft.n_timeframes == 5)
        self.assertTrue(np.allclose(ft.frames[ft.n_timeframes-1], SE3element().representation))

    def test_len(self):
        ft = FrameTrajectory("blargh", 100)
        self.assertTrue(len(ft) == 100)

#    def test_iter(self):
#        ft = FrameTrajectory("blargh", 100)
#        for frame in ft:
#            self.assertTrue(frame == SE3element())
#
    def test_add(self):
        len1 = 5
        len2 = 10
        ft1 = FrameTrajectory("blargh", len1)
        ft2 = FrameTrajectory("blorgh", 0)
        for _ in range(len2):
            ft2.append(SE3element())

        ft3 = ft1+ft2
        #print('ft1', (ft1.frames).shape)
        #print('ft2', (ft2.frames).shape)
        #print('ft3', (ft3.frames).shape)
        self.assertTrue(len(ft3) == len1+len2)
        self.assertTrue(((ft3).frames).shape == (16,4,4))
        self.assertTrue((ft3).n_timeframes == len1+len2)
        self.assertTrue((ft3).name == "blargh+blorgh")

    def test_trajmul(self):
        len1 = 10
        ft1 = FrameTrajectory("blargh", 0)
        for i in range(len1):
            ft1.append(SE3element())

        ft2 = FrameTrajectory("blorgh", 0)
        for i in range(len(ft1)):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            ft2.append(v)

        #print(ft1.frames[:ft1.n_timeframes,:,:].shape, ft2.frames[:ft2.n_timeframes,:,:].shape)
        ft3 = ft1*ft2
        #print(ft3.frames[:ft3.n_timeframes,:,:].shape)
        self.assertTrue(len(ft3) == 10)
        #ft3[-1] = SE3element()
        #ft2[-1] = SE3element()
        self.assertTrue(np.allclose(ft3.frames[:ft3.n_timeframes], ft2.frames[:ft2.n_timeframes]))

    def test_getsetitem(self):
        ft1 = FrameTrajectory("blargh", 10)
        ft2 = FrameTrajectory("blorgh")
        for i in range(len(ft1)):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            ft2.append(v)
        self.assertTrue(len(ft1*ft2) == 10)
        #print(ft2.all_frames)
        ft2[-1] = SE3element()
        ft2[0] = SE3element()
        #print(ft2.all_frames)
        self.assertTrue(ft2[0] == SE3element())
        self.assertTrue(ft2[-1] == SE3element())

    def test_getslice(self):
        ft1 = FrameTrajectory("blargh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            ft1.append(v)

        ft2 = ft1[:]
        self.assertTrue(ft1.frames.shape == ft2.frames.shape)
        self.assertTrue(ft1.all_frames.shape == ft2.all_frames.shape)
        ft2 = ft1[5:]
        self.assertTrue(ft2[-1] == ft1[-1])
        ft2 = ft1[:5]
        self.assertTrue(ft2[0] == ft1[0])
        ft2 = ft1[:-1]
        self.assertTrue(len(ft2) == len(ft1)-1)
        self.assertTrue(ft2[-1] == ft1[-2])

    def test_setslice(self):
        ft1 = FrameTrajectory("blargh")
        ft2 = FrameTrajectory("blargh")
        for i in range(20):
            ft1.append(SE3element(SO3.uniform_random(), np.random.rand(3)))
        for i in range(10):
            ft2.append(SE3element(SO3.uniform_random(), np.random.rand(3)))

        ft1[:10] = ft2
        self.assertTrue(ft1[:10].all_frames.shape == ft2.all_frames.shape)
        self.assertTrue(np.allclose(ft1.all_frames[:10], ft2.all_frames))
        ft1[10:] = ft2
        self.assertTrue(ft1[10:].all_frames.shape == ft2.all_frames.shape)
        self.assertTrue(np.allclose(ft1.all_frames[10:], ft2.all_frames))
        #print(ft1.all_frames)
        #print(ft2.all_frames)

    def test_positions(self):
        ft = FrameTrajectory("blargh")
        for _ in range(1):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            ft.append(v)
        print()
        print()
        print(np.transpose(ft.positions, (0,2,1)))
        print(json.dumps(ft.toJSON(), indent=4))
        #...

    def test_inverse(self):
        ft1 = FrameTrajectory("blorgh", 0)
        ft2 = FrameTrajectory("blargh", 0)
        for _ in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            ft1.append(v)
            ft2.append(v.inv())

        ft3 = ft1.inv()

        ##print(ft2.all_frames)
        #print(ft3.all_frames)
        self.assertTrue(np.allclose(ft2.all_frames, ft3.all_frames))

    def test_left_multiply(self):
        ft1 = FrameTrajectory("blorgh")
        for _ in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            #ft1.append(v)
            ft1.append(SE3element())

        w = SE3element()
        w.orientation = SO3.uniform_random()

        ft2 = ft1.left_multiply(w)

        longw = np.array([w.representation for i in range(10)])
        self.assertTrue(np.allclose(ft2.frames[:ft2.n_timeframes], longw[:ft2.n_timeframes]))

    def test_right_multiply(self):
        ft1 = FrameTrajectory("blorgh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            #ft1.append(v)
            ft1.append(SE3element())

        w = SE3element()
        w.orientation = SO3.uniform_random()

        ft2 = ft1.right_multiply(w)

        longw = np.array([w.representation for i in range(10)])
        self.assertTrue(np.allclose(ft2.frames[:ft2.n_timeframes], longw[:ft2.n_timeframes]))

    def test_json(self):
        print()
        pass

    def test_relative_to(self):
        ft1 = FrameTrajectory("blargh")
        ft2 = FrameTrajectory("blorgh")
        frames1 = []
        frames2 = []
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            ft1.append(v)
            frames1.append(v)
            w = SE3element()
            w.orientation = SO3.uniform_random()
            ft2.append(w)
            frames2.append(w)

        rel_ft = ft1.relative_to(ft2)

        self.assertTrue(np.allclose(rel_ft.frames, ft2.inv()*ft1))

    def test_log(self):
        ft = FrameTrajectory("blargh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            ft.append(v)

        all_logs =  np.array([np.real(sl.logm(ft.all_frames[i])) for i in range(ft.all_frames.shape[0])])
        #print(all_logs)
        #print(logSE3(ft))
        #print(np.matmul(ft.orientations, ft.positions))
        #print(np.matmul(np.transpose(ft.orientations, (0, 2, 1)), ft.positions))
        or_logs =  np.array([np.real(sl.logm(ft.orientations[i])) for i in range(ft.orientations.shape[0])])
        log_ft = logSO3(ft.orientations)


        self.assertTrue(np.allclose(log_ft, or_logs))

    def test_exp(self):
        ft = FrameTrajectory("blargh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            ft.append(v)

        log_ft = logSO3(ft.orientations)
        ors =  np.array([np.real(sl.expm(log_ft[i])) for i in range(log_ft.shape[0])])
        or_logexp = expSO3(log_ft)

        #print(or_logexp)
        #print(ft.orientations)
        #print(ors)

        self.assertTrue(np.allclose(ft.orientations, or_logexp))

    def test_sqrt(self):
        ft = FrameTrajectory("blargh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            ft.append(v)

        sqrt_ft = sqrt(ft.orientations)
        or2 = np.matmul(sqrt_ft,sqrt_ft)

        #print(or2)
        #print(ft.orientations)
        self.assertTrue(np.allclose(ft.orientations, or2))

    def test_midframe(self):
        ft1 = FrameTrajectory("blargh")
        ft2 = FrameTrajectory("blorgh")
        ft3 = FrameTrajectory("blorgh")
        frames1 = []
        frames2 = []
        frames3 = []
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            ft1.append(v)
            frames1.append(v)
            w = SE3element()
            w.orientation = SO3.uniform_random()
            ft2.append(w)
            frames2.append(w)
            ft3.append(SE3.midframe(v,w))

        mid_ft = midframes(ft1, ft2)
        #print(ft3.all_frames)
        #print(mid_ft.all_frames)

        self.assertTrue(np.allclose(ft3.all_frames, mid_ft.all_frames))

    def test_dof(self):
        ft1 = FrameTrajectory("blargh")
        ft2 = FrameTrajectory("blorgh")
        for i in range(10):
            v = SE3element()
            v.orientation = SO3.uniform_random()
            v.position = np.random.rand(3)
            #print('R1')
            #print(v.orientation)
            #print('r1')
            #print(v.position)
            ft1.append(v)
            w = SE3element()
            w.orientation = SO3.uniform_random()
            w.position = np.random.rand(3)
            #print('R2')
            #print(w.orientation)
            #print('r2')
            #print(w.position)
            ft2.append(w)

        dof_trajs = degrees_of_freedom(ft1, ft2)

        #print("dof_trajs")
        #print(dof_trajs)

        #self.assertTrue(np.allclose(dofs, dof_trajs))
#
#    def test_symmetrise(self):
#        ft1 = FrameTrajectory("blargh")
#        ft2 = FrameTrajectory("blorgh")
#        frames1 = []
#        frames2 = []
#        for i in range(10):
#            v = SE3element()
#            v.orientation = SO3.uniform_random()
#            ft1.append(v)
#            frames1.append(v)
#        for i in range(10):
#            v = SE3element()
#            v.orientation = SO3.uniform_random()
#            ft2.append(v)
#            frames2.append(v)
#
#        symm_ft = symmetrise(ft1, ft2)
        #check whether this is correct..

    def test_similarity_w_MCsim(self):
        R1 = [[ 0.618495667297936,  0.756520080223490, 0.212462885587871],
              [-0.757566881204802,  0.502261500717666, 0.416924220210901],
              [ 0.208699616758897, -0.418820669398589, 0.883761119788151]]

        R2 = [[ 0.501393685741601,  0.860550744865617, 0.089759609010310],
              [-0.737150991142028,  0.370556921874813, 0.565062814126881],
              [ 0.453004181169202, -0.349485311797069, 0.820150735341575]]

        r1 = [ 1.6184956672979363,  0.7565200802234899,  2.2124628855878708]

        r2 = [ 1.4530041811692018, -0.3494853117970693,  2.8201507353415751]

        rel_rot_mat =\
             [[ 0.963092798578408, 0.178587804940930, -0.201391800361361],
              [-0.180655686103367, 0.983512066456107,  0.008218163648920],
              [ 0.199538929547424, 0.028467719642060,  0.979476290950143]]

        rel_rot_axis =\
        [0.010251387066283,
        -0.202972159016950,
         -0.181867892757027]
        half_rot_axis =\
        [-0.092926126414646,
         -0.092768170944354,
         -0.036789688247863]
        midframe_or =\
        [[ 0.565431329375873,  0.811359160634517, 0.148269093929533],
         [-0.754122024769177,  0.435754243450524, 0.491343272135579],
         [ 0.334046978010982, -0.389633868868343, 0.858252914188130]]

        ft1 = FrameTrajectory("blargh")
        v = SE3element()
        v.orientation = SO3element(np.array(R1))
        v.position = np.array(r1)
        ft1.append(v)

        ft2 = FrameTrajectory("blorgh")
        w = SE3element()
        w.orientation = SO3element(np.array(R2))
        w.position = np.array(r2)
        ft2.append(w)

        #print()
        #print(ft2.relative_to(ft1).orientations)
        #print((ft1.inv()*ft2).orientations)

        #print('midframes(ft1, ft2).orientations')
        #print(midframes(ft1, ft2).orientations)

        #print('dof(ft1, ft2)')
        #print(degrees_of_freedom(ft1, ft2))


