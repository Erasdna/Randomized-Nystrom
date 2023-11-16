#(Parallel?) Sketching matrix generation

#Parallel SRHT

from mpi4py import MPI
import numpy as np
from scipy.linalg import hadamard

def SRHT(l,dim,R_seed,D_seed):
    #Generate R (dim x l) (should be the same everywhere)
    R_rng = np.random.default_rng(seed=R_seed)
    cols = R_rng.choice(dim,l)
    #Generate H (dim x dim) and extract columns (=H@R)
    H = (1/np.sqrt(dim))*hadamard(dim)[:,cols]

    #Sample Dl (dim x dim) and Dr (l x l)
    D_rng = np.random.default_rng(seed=D_seed)
    Dr = D_rng.choice([-1,1],l)
    Dl = D_rng.choice([-1,1],dim)

    # return sqrt(dim/l)* Dl @ HR @ Dr on all processes
    return np.sqrt(dim/l)*Dl[:,None] *(H * Dr[None,:])
