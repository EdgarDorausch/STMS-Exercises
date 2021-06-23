from Exercise06.datastructures import ParticleMethod

import numpy as np
import matplotlib.pyplot as plt

N = 50
X = 4

ny = 0.01
dt = 0.1
T  = 10.0

def exact(x, t):
    return x/np.power(1+4*ny*t, 3/2)*np.exp((-x**2)/(1+4*ny*t))

class RandomWalk(ParticleMethod):

    def __init__(self, N: int):
        self.h = 2*X/(2*N-1)
        x = np.linspace(-X, X, 2*N)[None,:]
        m = x*np.exp(-x**2)*self.h

        super().__init__(x, m, symmetric=True, reflexive=False)

    def evolve(self, t: int):
        dx_dt = np.random.normal(0.0, 2*ny, size=[1, self.NUM_PART])
        self.position_list += dx_dt*dt
        # print(self.position_list[:,0])

    def interact(self, p_idx, q_idx, t: int):
        pass

    def get_bins(self, num_bins):
        bin_width = X/num_bins

        bin_pos     = np.arange(start=-num_bins, stop=num_bins, step=1.0)*bin_width + bin_width/2
        x           = np.linspace(start=-X, stop=X, num=2*N)
        bin_vals      = np.zeros([2*num_bins])

        for i in range(self.NUM_PART):
            bin_idx = np.floor((self.position_list[:,i]+X-bin_width/2)/bin_width).astype(int)
            bin_vals[bin_idx] += self.properties_list[:,i]/bin_width
        
        return bin_pos, bin_vals


def main():
    rw = RandomWalk(100)
    plt.plot(*rw.get_bins(9), label='random walk init')
    rw.run(int(T/dt))
    plt.plot(*rw.get_bins(9), label='random walk')


    x = np.linspace(start=-X, stop=X, num=2*N)

    plt.plot(x, exact(x, T), label='exact')
    plt.plot(x, exact(x, 0.0), label='init')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()