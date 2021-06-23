from __future__ import annotations
import time
from typing import *
if TYPE_CHECKING:
    pass

import numpy as np
from abc import ABC, abstractmethod
import itertools


fill_empty_list = np.vectorize(lambda x: [], otypes=[object])


class ParticleMethod(ABC):

    def __init__(
        self,
        position_list: np.ndarray, # Shape [POS_DIM, NUM_PART]
        property_list: np.ndarray, # Shape [PROP_DIM, NUM_PART]
        symmetric: bool = False,
        reflexive: bool = True
    ):
        self.position_list = position_list
        self.properties_list = property_list
        self.symmetric = symmetric
        self.reflexive = reflexive

        self.POS_DIM, self.NUM_PART = position_list.shape
        self.PROP_DIM = property_list.shape[0]
        assert self.NUM_PART == property_list.shape[1], f'Length missmatch: {self.NUM_PART} != {property_list.shape[1]}'


    @abstractmethod
    def evolve(self, t: int):
        """
        Inplace evolve logic
        """
        raise NotImplementedError()

    @abstractmethod
    def interact(self, p_idx, q_idx, t: int):
        """
        Inplace interact logic
        """
        raise NotImplementedError()

    def _interacting_particles(self) -> Generator[Tuple[int, int], None, None]:
        for p_idx in range(self.NUM_PART):
                
            q_range = None
            if self.symmetric and self.reflexive:
                q_range = range(p_idx+1)
            elif self.symmetric and not self.reflexive:
                q_range = range(p_idx)
            elif not self.symmetric and self.reflexive:
                q_range = range(self.NUM_PART)
            elif not self.symmetric and not self.reflexive:
                q_range = itertools.chain(range(p_idx), range(p_idx+1, self.NUM_PART))
            else:
                raise Exception('Unhandled case')

            for q_idx in q_range:
                yield p_idx, q_idx

    def _step(self, t: int):
        for p_idx, q_idx in self._interacting_particles():
            self.interact(p_idx, q_idx, t)
        self.evolve(t)
        self._after_step(t)

    def _after_step(self, t: int):
        pass
        
    def run(self, T: int):
        print('Running particle method...')
        t0 = time.time()
        for t in range(T):
            self._step(t)
        t1 = time.time()
        print(f'... it took {t1-t0:.3f}s')




class CellList(ParticleMethod):

    def __init__(self,
        position_list: np.ndarray, # Shape [POS_DIM, NUM_PART]
        property_list: np.ndarray, # Shape [PROP_DIM, NUM_PART]
        pos_min: np.ndarray, # Shape [POS_DIM]
        pos_max: np.ndarray, # Shape [POS_DIM]
        r_cutoff: float,
        symmetric: bool = False,
        reflexive: bool = True
    ):
        super().__init__(position_list, property_list, symmetric, reflexive)
        self.r_cutoff = r_cutoff
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.grid_res = np.ceil((pos_max-pos_min)/r_cutoff).astype(int)
        
        # create numpy array containing the cell lists
        self.cell_list = np.empty(shape=self.grid_res, dtype=object)
        self._construct_cell_list()
        

    def _construct_cell_list(self):
        self.cell_list = fill_empty_list(self.cell_list)

        # fill arrays
        for p_idx in range(self.NUM_PART):
            cell_idx = tuple(x for x in np.nditer(np.floor((self.position_list[:,p_idx]-self.pos_min)/self.r_cutoff).astype(int)))
            self.cell_list[cell_idx].append(p_idx)

    def get_adjacent_cells(self, cell_index: Sequence[int]):
        # generate all neighbors along one dimension
        def dim_neighbors(neighbors: List[Tuple[int,...]], dim: int):
            for neighbor in neighbors:
                x = neighbor[0:dim]
                for delta in [1,-1]:
                    if not 0 <= neighbor[dim]+delta < self.grid_res[dim]: # If not in grid
                        continue
                    y = x + (neighbor[dim]+delta,) 
                    if dim < len(neighbor): # If last dimension
                        y += neighbor[dim+1:]
                    yield y


        neighbor_list = [tuple(cell_index)]
        for dim in range(len(cell_index)):
            neighbor_list.extend(list(dim_neighbors(neighbor_list, dim)))
        
        return neighbor_list[1:] # dont return current cell

    def get_cell_index(self, particle_index: int) -> Tuple[int, ...]:
        return tuple(x for x in np.nditer(np.floor((self.position_list[:, particle_index]-self.pos_min)/self.r_cutoff).astype(int)))

    def get_particle_distance(self, p_idx, q_idx):
        diff = self.position_list[:,p_idx] - self.position_list[:,q_idx]
        return np.linalg.norm(diff)

    @abstractmethod
    def evolve(self, t: int):
        """
        Inplace evolve logic
        """
        raise NotImplementedError()

    @abstractmethod
    def interact(self, p_idx, q_idx, t: int):
        """
        Inplace interact logic
        """
        raise NotImplementedError()


    def _interacting_particles(self) -> Generator[Tuple[int, int], None, None]:
        for p_idx in range(self.NUM_PART):
                
            q_range = None
            if self.symmetric and self.reflexive:
                raise NotImplementedError('Symmetric case not implemented')

            elif self.symmetric and not self.reflexive:
                raise NotImplementedError('Symmetric case not implemented')

            elif not self.symmetric and self.reflexive:
                p_cell = self.get_cell_index(p_idx)
             
                for q_idx in self.cell_list[p_cell]:
                    yield p_idx, q_idx

                for adj_cell in self.get_adjacent_cells(p_cell):
                    for q_idx in self.cell_list[adj_cell]:
                        if self.get_particle_distance(p_idx, q_idx) < self.r_cutoff:
                            yield p_idx, q_idx
           
            elif not self.symmetric and not self.reflexive:
                p_cell = self.get_cell_index(p_idx)
             
                for q_idx in self.cell_list[p_cell]:
                    yield p_idx, q_idx

                for adj_cell in self.get_adjacent_cells(p_cell):
                    for q_idx in self.cell_list[adj_cell]:
                        if self.get_particle_distance(p_idx, q_idx) < self.r_cutoff and p_idx != q_idx:
                            yield p_idx, q_idx
            else:
                raise Exception('Unhandled case')

    def _after_step(self, t: int):
        self._construct_cell_list()



class VerletList(CellList):
    def __init__(self,
        position_list: np.ndarray, # Shape [POS_DIM, NUM_PART]
        property_list: np.ndarray, # Shape [PROP_DIM, NUM_PART]
        pos_min: np.ndarray, # Shape [POS_DIM]
        pos_max: np.ndarray, # Shape [POS_DIM]
        r_cutoff: float,
        r_skin: float,
        symmetric: bool = False,
        reflexive: bool = True
    ) -> None:
        super().__init__(position_list, property_list, pos_min, pos_max, r_cutoff+r_skin, symmetric, reflexive)

        NUM_PART = position_list.shape[1]

        self.r_skin = r_skin
        self.r_orig_cutoff = r_cutoff

        self._construct_verlet_list()

        
    def _construct_verlet_list(self):

        print(f'Constructing Verlet List for {self.NUM_PART} particles')
        t0 = time.time()
        fill_empty_list = np.vectorize(lambda x: [], otypes=[object])
        self.verlet_list = fill_empty_list(np.empty(shape=[self.NUM_PART], dtype=object))

        for particle_index in range(self.NUM_PART):

            own_cell_index =  self.get_cell_index(particle_index)
            own_cell_list = self.cell_list[own_cell_index]
            self.verlet_list[particle_index].extend([
                q_idx for q_idx in own_cell_list if self.get_particle_distance(q_idx, particle_index) <= self.r_cutoff
                ])

            for neighbor_cell_index in self.get_adjacent_cells(own_cell_index):
                neighbor_cell_list = self.cell_list[neighbor_cell_index]
                self.verlet_list[particle_index].extend([
                    q_idx for q_idx in neighbor_cell_list if self.get_particle_distance(q_idx, particle_index) < self.r_cutoff
                    ])

        t1 = time.time()
        print(f'Verlet List construction took ~{t1-t0:.5}s')


    def _interacting_particles(self) -> Generator[Tuple[int, int], None, None]:
        for p_idx in range(self.NUM_PART):
                
            q_range = None
            if self.symmetric and self.reflexive:
                raise NotImplementedError('Symmetric case not implemented')

            elif self.symmetric and not self.reflexive:
                raise NotImplementedError('Symmetric case not implemented')

            elif not self.symmetric and self.reflexive:
                for q_idx in self.verlet_list[p_idx]:
                    yield p_idx, q_idx
           
            elif not self.symmetric and not self.reflexive:
                for q_idx in self.verlet_list[p_idx]:
                    if p_idx != q_idx:
                        yield p_idx, q_idx
            else:
                raise Exception('Unhandled case')


    @abstractmethod
    def _check_update(self) -> bool:
        raise NotImplementedError() 
    
    def _after_step(self, t: int):
        if self._check_update():
            self._construct_cell_list()
            self._construct_verlet_list()