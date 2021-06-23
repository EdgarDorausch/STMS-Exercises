from Exercise06.datastructures import ParticleMethod

import numpy as np
import matplotlib.pyplot as plt

N = 50
X = 4

ny = 0.0001
dt = 0.1
T  = 50.0

def exact(x, t):
    return x/np.power(1+4*ny*t, 3/2)*np.exp((-x**2)/(1+4*ny*t))

class RandomWalk(ParticleMethod):

    def __init__(self, N: int):
        self.h = 2*X/(2*N-1)
        x = np.linspace(-X, X, 2*N)[None,:]
        m = x*np.exp(-x**2)*self.h

        super().__init__(x, m, symmetric=True, reflexive=False)

    def evolve(self, t: int):
        dx_dt = np.random.normal(0.0, 2*ny)
        self.position_list += dt * dx_dt
        # print(self.position_list[:,0])

    def interact(self, p_idx, q_idx, t: int):
        pass



if __name__ == "__main__":
    rw = RandomWalk(50)
    rw.run(int(T/dt))

    print(rw.position_list)

    NUM_BINS = 25
    BIN_WIDTH = X/NUM_BINS
    x = np.arange(start=-NUM_BINS, stop=NUM_BINS, step=1.0)*BIN_WIDTH + BIN_WIDTH/2
    bins = np.zeros([2*NUM_BINS])
    for i in range(rw.NUM_PART):
        bin_idx = np.floor((rw.position_list[:,i]+X-BIN_WIDTH/2)/BIN_WIDTH).astype(int)
        bins[bin_idx] += rw.properties_list[:,i]/rw.h
    # for i, curr_x in enumerate(x):
    #     mask = curr_x-X/N < rw.position_list
    #     mask = np.logical_and(mask, rw.position_list <= curr_x+X/N)
    #     print(i,np.sum(mask))
    #     u[i] = np.sum(rw.properties_list[mask])/rw.h

    plt.plot(x,bins, label='random walk')
    plt.plot(x, exact(x, T), label='exact')
    plt.plot(x, exact(x, 0.0))
    plt.legend()
    plt.show()