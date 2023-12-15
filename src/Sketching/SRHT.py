import numpy as np
import torch
from hadamard_transform import hadamard_transform, pad_to_power_of_2
from .Sketch import Sketch

class SRHT(Sketch):
    def __init__(self, l, dim, base_seed) -> None:
        super().__init__(l, dim, base_seed)
        self.cols = self.base_rng.choice(self.dim,self.l,replace=False)

    def initialize_matrix(self, seed):
        super().initialize_matrix(seed)
        self.Dl = self.process_rng.choice([-1,1],self.l)
        self.Ddim = self.process_rng.choice([-1,1],self.dim)
    
    def apply_right(self,A):
        super().apply_right(A)
        tmp = ((fht(A * self.Ddim[:,None]))[:,self.cols])
        return tmp*self.Dl[None,:]*np.sqrt(self.dim/self.l)

def fht(mat):
    #Fast Walsh-Hadamard using pytorch
    return hadamard_transform(pad_to_power_of_2(torch.from_numpy(mat))).numpy()