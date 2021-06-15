from Exercise06.datastructures import ParticleMethod

import numpy as np
import matplotlib.pyplot as plt

N = 200
X = 4

ny = 0.0001
dt = 0.1
T  = 500.0

def exact(x, t):
    return x/np.power(1+4*ny*t, 3/2)*np.exp((-x**2)/(1+4*ny*t))

class RandomWalk(ParticleMethod):

    def __init__(self, N: int):
        x = np.linspace(-X, X, N)[None,:]
        u = x*np.exp(-x**2)

        super().__init__(x, u, symmetric=True, reflexive=False)

    def evolve(self, t: int):
        dx_dt = np.random.normal(0.0, 2*ny)
        self.position_list += dt * dx_dt

    def interact(self, p_idx, q_idx, t: int):
        pass



if __name__ == "__main__":
    rw = RandomWalk(N)
    rw.run(int(T/dt))

    x = np.linspace(-X, X, N)
    u = np.empty(x.shape)
    for i, curr_x in enumerate(x):
        mask = curr_x-X/N < rw.position_list
        mask = np.logical_and(mask, rw.position_list <= curr_x+X/N)
        u[i] = np.sum(rw.properties_list[mask])

    plt.plot(x,u, label='random walk')
    plt.plot(x, exact(x, T), label='exact')
    plt.plot(x, exact(x, 0.0))
    plt.legend()
    plt.show()