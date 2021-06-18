from Exercise06.datastructures import ParticleMethod, VerletList

import numpy as np
import matplotlib.pyplot as plt

N = 200
X = 4

ny = 0.0001
dt = 0.1
T  = 10.0


def exact(x, t):
    return x/np.power(1+4*ny*t, 3/2)*np.exp((-x**2)/(1+4*ny*t))

def eta_eps(x_diff: np.ndarray, epsilon: float):
    return 1/(2*epsilon*np.sqrt(np.pi))*np.exp(-(x_diff/(2*epsilon))**2)


class PSE(VerletList):

    def __init__(self, N: int):
        self.h = 2*X/(2*N-1)
        self.epsilon = self.h

        x = np.linspace(-X, X, 2*N)[None,:]
        prop = np.empty([2,2*N])
        prop[0,:] = exact(x, 0.0)*self.h
        prop[1,:] = 0.0

        super().__init__(x, prop, pos_min=np.array([-X]), pos_max=np.array([X]), r_cutoff=4*self.epsilon, r_skin=0.0, symmetric=False, reflexive=False)

    def evolve(self, t: int):
        self.properties_list[0] += dt * self.properties_list[1] * ny * self.h / self.epsilon**2 
        self.properties_list[1] = 0.0

    def interact(self, p_idx, q_idx, t: int):
        self.properties_list[1, p_idx] += (self.properties_list[0, q_idx]-self.properties_list[0, p_idx]) * \
            eta_eps(self.position_list[0,q_idx]-self.position_list[0,p_idx], self.epsilon)

    def _check_update(self) -> bool:
        return False # Particles not moving => No need for updating the Verlet List

if __name__ == "__main__":
    # rw = PSE(N)
    # rw.run(int(T/dt))

    # x = np.linspace(-X, X, 2*N)
    
    # plt.plot(rw.position_list[0], rw.properties_list[0]/rw.h, label='pse')
    # plt.plot(x, exact(x, T), label='exact')
    # plt.plot(x, exact(x, 0.0), label='start distribution')
    # plt.legend()
    # plt.show()

    Ns = [50, 100, 200, 400, 800]
    errors = []
    for N in Ns:
        rw = PSE(N)
        rw.run(int(T/dt))

        err = np.sqrt(np.mean( (exact(rw.position_list[0], T) - rw.properties_list[0]/rw.h)**2))
        errors.append(err)

    plt.plot(Ns, errors)
    plt.show()