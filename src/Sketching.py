#(Parallel?) Sketching matrix generation

#Parallel SRHT

from mpi4py import MPI
import numpy as np
from scipy.linalg import hadamard
import time
import torch
from hadamard_transform import hadamard_transform, pad_to_power_of_2
import scipy.sparse as sp

def SRHT(A,l,dim,R_seed,D_seed):
    #Generate R (dim x l) (should be the same everywhere)
    R_rng = np.random.default_rng(seed=R_seed)
    #Sample columns without replacement
    cols = R_rng.choice(dim,l,replace=False)

    #Sample Dl (dim x dim) and Dr (l x l)
    D_rng = np.random.default_rng(seed=D_seed)
    Dr = D_rng.choice([-1,1],l)
    rad = D_rng.choice([-1,1],dim)
    #Dl = np.diag(rad) #np.zeros((l,dim))
    #Dl[np.arange(l),cols] = rad
    #Apply the fast Hadamard transform to the Dl matrix
    #DPH = (fht(Dl))[:,cols]
    hadamard_dim = 2**(np.ceil(np.log2(dim)).astype(int)) 
    DPH =((A*rad[:,None]) @ (hadamard(hadamard_dim)))[:dim,cols]
    ret = np.sqrt(1/l)*(DPH * Dr[None,:])
    return ret

def fht2(x):
    batch_dim = len(x)
    d = len(x)

    h = 2
    while h <= d:
        hf = h // 2
        cutoff = d // h 
        #x = np.reshape(x,(batch_dim, d // h, h))

        half_1, half_2 = x[:, :, 0,:], x[:, :, 1,:]

        x[:,:,0,:],x[:,:,1,:] = half_1 + half_2, half_1 - half_2
        h *= 2

    return (x / np.sqrt(d)).reshape((batch_dim,d))

def fht(mat):
    #Fast Walsh-Hadamard using pytorch
    return hadamard_transform(pad_to_power_of_2(torch.from_numpy(mat))).numpy()

def SASO(l,dim,R_seed,D_seed, d=8):
    d = np.min([d,l]) 
    R_rng = np.random.default_rng(seed=D_seed)
    Rademacher = R_rng.choice([1,-1],size=(dim,d))
    
    #Sample k non-zero elements per row
    col = R_rng.uniform(size=(dim,l)).argsort(1)[:,:d]
    row = np.indices((dim,d))[0]
    ret = np.zeros((dim,l))
    ret[row,col] = Rademacher
    return ret

def Gaussian(l,dim,R_seed,D_seed):
    rng = np.random.default_rng(seed=D_seed)

    #Sample N(0,1) Gaussian
    ret = rng.normal(loc=0,scale=1.0,size=(dim,l))
    return ret
        