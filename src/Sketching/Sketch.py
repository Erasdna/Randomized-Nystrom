import numpy as np

class Sketch:
    def __init__(self,l,dim,base_seed) -> None:
        self.base_rng = np.random.default_rng(seed=base_seed)
        self.base_seed = base_seed
        self.l=l 
        self.dim=dim
        self.initialized=False
    
    def initialize_matrix(self,seed):
        self.initialized=True
        self.process_seed = self.base_seed + seed
        self.process_rng = np.random.default_rng(seed=self.process_seed)
        pass
    def apply_right(self,A):
        assert self.initialized
        pass 
    def apply_left(self,A):
        assert self.initialized
        #Omega^T @ A = (A^T Omega)^T
        return self.apply_right(A.T).T