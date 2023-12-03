#(Parallel?) Sketching matrix generation

#Parallel SRHT

from mpi4py import MPI
import numpy as np
from scipy.linalg import hadamard
import time
from sympy import fwht
import scipy.sparse as sp

def SRHT(l,dim,R_seed,D_seed):
    t0 = time.perf_counter()
    #Generate R (dim x l) (should be the same everywhere)
    R_rng = np.random.default_rng(seed=R_seed)
    #Sample columns without replacement
    cols = R_rng.choice(dim,l,replace=False)

    #Generate H (dim x dim) and extract columns (=H@R)
    hadamard_dim = 2**(np.ceil(np.log2(dim)).astype(int))
    start = (hadamard_dim - dim)//2
    stop = dim + start
    H = (hadamard(hadamard_dim))[start:stop,cols]

    #Sample Dl (dim x dim) and Dr (l x l)
    D_rng = np.random.default_rng(seed=D_seed)
    Dr = D_rng.choice([-1,1],l)
    Dl = D_rng.choice([-1,1],dim)

    # return sqrt(dim/l)* Dl @ HR @ Dr on all processes
    ret = np.sqrt(1/l)*Dl[:,None] *(H * Dr[None,:])
    #print("SRHT sketch: ", time.perf_counter() - t0)
    return ret

def SASO(l,dim,R_seed,D_seed, fac=0.5):
    k = int(l*fac) 
    R_rng = np.random.default_rng(seed=R_seed)
    Rademacher = R_rng.choice([1,-1],size=k*dim)
    col = np.array([R_rng.choice(l,k,replace=False) for i in range(dim)]).flatten()
    row = np.repeat(np.arange(dim),k)
    ret = sp.csr_matrix((Rademacher,(row,col)),shape=(dim,l))
    return ret.todense()

def Gaussian(l,dim,R_seed,D_seed):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed=D_seed)
    ret = rng.normal(loc=0,scale=1.0,size=(dim,l))
    #print("Gaussian sketch: ", time.perf_counter() - t0)
    return ret