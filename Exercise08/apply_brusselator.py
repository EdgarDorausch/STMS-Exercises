import numpy as np
import matplotlib.pyplot as plt


def apply_brusselator(uv: np.ndarray, a: float, b: float, k: float) -> np.ndarray:
    """
    == param ==
        uv - shape: [2, NUM_PARTICLE]
    """
    duv_dt = np.empty(uv.shape)
    ku2v = k * uv[0]**2 * uv[1]

    duv_dt[0] = a+ku2v-(b+1)*uv[0]
    duv_dt[1] = b*uv[0]-ku2v

    return duv_dt



def main():
    uv = np.array([[0.7],[0.04]])
    a = 2.0
    b = 6.0
    k = 1.0

    T = 20
    dt = 0.01

    N_it = int(T/dt)
    t = np.arange(start=0.0, stop=T, step=dt)
    uv_list = np.empty([2,N_it])

    for it in range(N_it):
        uv_list[:, it] = uv[:,0]
        duv_dt = apply_brusselator(uv, a, b, k)
        uv += dt * duv_dt

    plt.plot(t, uv_list[0], label='u')
    plt.plot(t, uv_list[1], label='v')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()