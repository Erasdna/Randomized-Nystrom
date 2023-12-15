import numpy as np
from .Sketch import Sketch

class SASO(Sketch):
    def __init__(self, l, dim, base_seed,d=8) -> None:
        super().__init__(l, dim, base_seed)
        assert d<l 

        self.d=d
        self.rows = np.indices((dim,d))[0]
    
    def initialize_matrix(self, seed):
        super().initialize_matrix(seed)
        self.rademacher = self.process_rng.choice([1,-1],size=(self.dim,self.d))
        self.cols = self.process_rng.uniform(size=(self.dim,self.l)).argsort(1)[:,:self.d]
        self.saso = np.zeros((self.dim,self.l))
    
    def apply_right(self, A):
        super().apply_right(A)
        return A @ self.saso