import numpy as np
import matplotlib.pyplot as plt

from Exercise08.apply_brusselator import apply_brusselator
from Exercise06.datastructures import VerletList


def gauss2d_kernel(x_diff: np.ndarray, epsilon: float):
    return 1/(2*epsilon*np.sqrt(np.pi))*np.exp(-(np.linalg.norm(x_diff)/(2*epsilon))**2)

class BrusselatorDiffusion(VerletList):
    
    def __init__(self, a, b, k, D, dt):
        self.a = a
        self.b = b
        self.k = k
        self.dt = dt
        self.D = D

        pos = np.mgrid[0:51, 0:51] * 81/51 # [2, 51, 51]
        x = pos.reshape([2, -1])

        self.h = 51/81
        self.epsilon = self.h

        prop = np.zeros([4, 51*51])
        prop[0:2] = np.random.uniform(size=[2,51*51])
        prop[1,:] += 7.0

        super().__init__(x, prop, pos_min=np.array([0.0, 0.0]), pos_max=np.array([81.0, 81.0]), r_cutoff=4*self.epsilon, r_skin=0.0, symmetric=False, reflexive=False)

    def evolve(self, t: int):
        self.properties_list[0:2] += self.dt * self.properties_list[2:4] * self.D * self.h / self.epsilon**2 
        self.properties_list[2:4] = 0.0

        # TODO: add reaction term 

    def interact(self, p_idx, q_idx, t: int):
        self.properties_list[2:4, p_idx] += (self.properties_list[0:2, q_idx]-self.properties_list[0:2, p_idx]) * \
            gauss2d_kernel(self.position_list[0:2,q_idx]-self.position_list[0:2,p_idx], self.epsilon)

    def _check_update(self) -> bool:
        return False # Particles not moving => No need for updating the Verlet List



def main():
    a = 2.0
    b = 6.0
    k = 1.0
    D = 10.0

    T = 10
    dt = 0.01
    
    bru_diff = BrusselatorDiffusion(a,b,k,D,dt)
    bru_diff.run(int(10)) # TODO: Adjust iteration number

if __name__ == '__main__':
    main()