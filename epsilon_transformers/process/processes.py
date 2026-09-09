import numpy as np
from typing import Optional, cast
from types import FrameType
import inspect

from epsilon_transformers.process.Process import Process,NormTransitionMixin

class ZeroOneR(Process):
    def __init__(self, prob_of_zero_from_r_state: float = 0.5,**kwargs):
        self.name = "z1r"
        self.p = prob_of_zero_from_r_state
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        state_names = {"0": 0, "1": 1, "R": 2}
        T[0, state_names["0"], state_names["1"]] = 1.0
        T[1, state_names["1"], state_names["R"]] = 1.0
        T[0, state_names["R"], state_names["0"]] = self.p
        T[1, state_names["R"], state_names["0"]] = 1 - self.p

        return T, state_names


class Trun_Mess3(Process):
    def __init__(self, x=0.15, a=0.6,r=1,t1=1,t2=2,**kwargs):
        self.name = "trun_mess3"
        self.x = x
        self.a = a
        self.r = r
        self.t1 = t1
        self.t2 = t2
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x


        T[0, :, :] = [[0, 0, bx/2], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by+ay, ax+bx, 1.5*bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]


        return T,state_names   

class Linear_Mess3(NormTransitionMixin,Process):
    def __init__(self, x=0.15, a=0.6,**kwargs):
        self.name = "linear_mess3"
        self.x = x
        self.a = a
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]


        return T,state_names
    
    def _create_norm_matrix(self):
        
        T_n = np.zeros((3,3,3))
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        ay2bx=ay+2*bx
        axbybx=ax+by+bx

        T_n[0, :, :] = [[ay/ay2bx, bx/ay2bx, bx/ay2bx], [ax/axbybx, by/axbybx, bx/axbybx], [ax/axbybx, bx/axbybx, by/axbybx]]
        T_n[1, :, :] = [[by/axbybx, ax/axbybx, bx/axbybx], [bx/ay2bx, ay/ay2bx, bx/ay2bx], [bx/axbybx, ax/axbybx, by/axbybx]]
        T_n[2, :, :] = [[by/axbybx, bx/axbybx, ax/axbybx], [bx/axbybx, by/axbybx, ax/axbybx], [bx/ay2bx, bx/ay2bx, ay/ay2bx]]

        return T_n
    
class Mess3(Process):
    def __init__(self, x=0.15, a=0.6,**kwargs):
        self.name = "mess3"
        self.x = x
        self.a = a
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names    
    
 
class TransitionMatrixProcess(Process):
    def __init__(self, transition_matrix: np.ndarray,**kwargs):
        self.transition_matrix = transition_matrix
        super().__init__(**kwargs)

    def _create_hmm(self):
        return self.transition_matrix, {
            i: i for i in range(self.transition_matrix.shape[0])
        }
    
PROCESS_REGISTRY: dict[str, type] = {
    key: value
    # cast because we know the current frame has the above classes
    for key, value in cast(FrameType, inspect.currentframe()).f_locals.items()
    if isinstance(value, type) and issubclass(value, Process) and key != "Process"
}
