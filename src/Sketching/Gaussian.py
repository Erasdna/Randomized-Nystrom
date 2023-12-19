import numpy as np
from .Sketch import Sketch

class Gaussian(Sketch):
    def __init__(self, l, dim, base_seed) -> None:
        super().__init__(l, dim, base_seed)
    
    def initialize_matrix(self, seed):
        super().initialize_matrix(seed)
        self.gaussian = self.process_rng.normal(loc=0.0,scale=1.0,size=(self.dim,self.l))
    
    def apply_right(self, A):
        super().apply_right(A)
        return A @ self.gaussian