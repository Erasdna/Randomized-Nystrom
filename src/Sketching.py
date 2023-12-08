#(Parallel?) Sketching matrix generation

#Parallel SRHT

from mpi4py import MPI
import numpy as np
from scipy.linalg import hadamard
import time
import torch
from hadamard_transform import hadamard_transform, pad_to_power_of_2
import scipy.sparse as sp

def SRHT(l,dim,R_seed,D_seed):
    #Generate R (dim x l) (should be the same everywhere)
    R_rng = np.random.default_rng(seed=R_seed)
    #Sample columns without replacement
    cols = R_rng.choice(dim,l,replace=False)

    #Sample Dl (dim x dim) and Dr (l x l)
    D_rng = np.random.default_rng(seed=D_seed)
    Dr = D_rng.choice([-1,1],l)
    Dl = np.diag(D_rng.choice([-1,1],dim))

    #Apply the fast Hadamard transform to the Dl matrix
    DPH = (fht(Dl[cols,:])[:,:dim]).T 

    ret = np.sqrt(dim/l)*(DPH * Dr[None,:])
    return ret

def fht(mat):
    #Fast Walsh-Hadamard using pytorch
    return hadamard_transform(pad_to_power_of_2(torch.from_numpy(mat))).numpy()

def SASO(l,dim,R_seed,D_seed, d=8):
    d = np.min([d,l]) 
    R_rng = np.random.default_rng(seed=R_seed)
    Rademacher = R_rng.choice([1,-1],size=d*dim)
    
    #Sample k non-zero elements per row
    col = np.array([R_rng.choice(l,d,replace=False) for i in range(dim)]).flatten()
    row = np.repeat(np.arange(dim),d)
    ret = sp.csr_matrix((Rademacher,(row,col)),shape=(dim,l))
    return ret

def Gaussian(l,dim,R_seed,D_seed):
    rng = np.random.default_rng(seed=D_seed)

    #Sample N(0,1) Gaussian
    ret = rng.normal(loc=0,scale=1.0,size=(dim,l))
    return ret