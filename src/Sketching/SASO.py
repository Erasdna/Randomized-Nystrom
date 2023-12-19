import numpy as np
from .Sketch import Sketch

class SASO(Sketch):
    def __init__(self, l : int , dim : int, base_seed : int,d : int=8) -> None:
        super().__init__(l, dim, base_seed)
        assert d<l 

        self.d=d
        #Build row indices
        self.rows = np.indices((dim,d))[0]
    
    def initialize_matrix(self, seed : int ):
        """Initialize the SASO sketching matrix

        Args:
            seed (int): Process seed
        """
        super().initialize_matrix(seed)
        #Sample Rademacher vector
        self.rademacher = self.process_rng.choice([1,-1],size=(self.dim,self.d))
        #Sample columns (process specific)
        self.cols = self.process_rng.uniform(size=(self.dim,self.l)).argsort(1)[:,:self.d]
        self.saso = np.zeros((self.dim,self.l))
        self.saso[self.rows,self.cols] = self.rademacher
    
    def apply_right(self, A):
        super().apply_right(A)
        return A @ self.saso
    