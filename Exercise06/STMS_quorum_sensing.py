import pandas as pd
import numpy as np
from datastructures import VerletList
import matplotlib.pyplot as plt


class MyParticleMethod(VerletList):
    def evolve(self):
        pass

    def interact(self, p_idx, q_idx, t: int):
        pass 

    def _check_update(self):
        return False



if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv('Exercise06/QSBacterialPos.dat', sep="\s+", header=None)
    print(df)

    # pl = [MyParticle(row) for index, row in df.iterrows()]

    print(df.shape)
    cl = MyParticleMethod(
        df.to_numpy().T,
        np.empty([0,df.shape[0]]),
        pos_min=df.min(),
        pos_max=df.max(),
        r_cutoff=0.1,
        r_skin=0.02
    )


    highlight_particle = 2

    print(cl.verlet_list[highlight_particle])

    print(cl.get_adjacent_cells([6,9]))

    get_len = np.vectorize(lambda x: len(x), otypes=[int])

    print(get_len(cl.cell_list)[::-1,:])

    cell = cl.cell_list[highlight_particle,1]
    verlet_neighbors = cl.verlet_list[2]

    colors = ['r' if i in verlet_neighbors else 'b'  for i in range(df.shape[0])]

    colors[highlight_particle] = 'g'

    circle2 = plt.Circle(cl.position_list[::-1,2], cl.r_cutoff, color='b', fill=False)

    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.scatter(y=cl.position_list[0,:], x=cl.position_list[1,:], c=colors)
    ax.set_xticks(np.arange(cl.pos_min[1], cl.pos_max[1], step=cl.r_cutoff))
    ax.set_yticks(np.arange(cl.pos_min[0], cl.pos_max[0], step=cl.r_cutoff))
    ax.grid(True)
    ax.add_patch(circle2)
    plt.show()
