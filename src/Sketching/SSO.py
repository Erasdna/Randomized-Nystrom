import numpy as np
from .Sketch import Sketch

class SSO(Sketch):
    def __init__(self, l : int, dim : int, base_seed : int,d : int=16) -> None:
        super().__init__(l, dim, base_seed)
        assert d<l 
        self.d=d
    
    def initialize_matrix(self, seed : int):
        """Initialize the SSO sketching matrix

        Args:
            seed (int): Process seed
        """
        super().initialize_matrix(seed)
        self.rademacher = self.process_rng.choice([1,-1],size=self.dim*self.d)
        self.cols = self.process_rng.choice(self.l,size=self.dim*self.d)
        self.rows = self.process_rng.choice(self.dim,size=self.dim*self.d)
        self.sso = np.zeros((self.dim,self.l))
        self.sso[self.rows,self.cols] = self.rademacher
    def apply_right(self, A):
        super().apply_right(A)
        return A @ self.sso
    