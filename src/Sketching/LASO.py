import numpy as np
from .Sketch import Sketch

class LASO(Sketch):
    def __init__(self, l : int , dim : int, base_seed: int,d : int =60) -> None:
        super().__init__(l, dim, base_seed)
        assert d<dim

        self.d=d
        #Make column indices
        self.cols = np.indices((d,l))[1]
    
    def initialize_matrix(self, seed : int ):
        """Initialize the LASO sketching matrix

        Args:
            seed (int): Process seed
        """
        super().initialize_matrix(seed)
        #Sample rademacher vector
        self.rademacher = self.process_rng.choice([1,-1],size=(self.d,self.l))
        #Sample row indices (process specific)
        self.rows = self.process_rng.uniform(size=(self.dim,self.l)).argsort(0)[:self.d,:]
        self.laso = np.zeros((self.dim,self.l))
        self.laso[self.rows,self.cols] = self.rademacher
    
    def apply_right(self, A):
        super().apply_right(A)
        return A @ self.laso
    