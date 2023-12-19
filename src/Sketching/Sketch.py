import numpy as np

class Sketch:
    """Sketch is the base Sketching class from which Gaussian, SASO, LASO and SSO are derived
    """
    def __init__(self,l : int,dim : int ,base_seed : int) -> None:
        """Initialize a sketching object

        Args:
            l (int): Oversampling dimension
            dim (int): Matrix dimension
            base_seed (int): Seed
        """
        self.base_rng = np.random.default_rng(seed=base_seed)
        self.base_seed = base_seed
        self.l=l 
        self.dim=dim
        self.initialized=False
    
    def initialize_matrix(self,seed : int):
        """Build the sketching matrix. We can assign seed for each process

        Args:
            seed (int): Process specific seed
        """
        self.initialized=True
        self.process_seed = self.base_seed + seed
        self.process_rng = np.random.default_rng(seed=self.process_seed)
        pass
    
    def apply_right(self,A):
        # A @ Omega
        assert self.initialized
        pass 
    def apply_left(self,A):
        assert self.initialized
        #Omega^T @ A = (A^T Omega)^T
        return self.apply_right(A.T).T