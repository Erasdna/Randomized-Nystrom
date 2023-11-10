from mpi4py import MPI
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from TSQR import TSQR

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def f(x,mu):
    return np.sin(10*(mu[None,:]+x[:,None]))/(np.cos(100*(mu[None,:]-x[:,None])) + 1.1)

m = 30000
n = 600
X = np.arange(m)/(m-1)
MU = np.arange(n)/(n-1)

mat = f(X,MU)

if rank==0:
    tot = mat.shape[0]
    row = tot//size
    col = mat.shape[1]
    W = np.reshape(mat,(size,row,col))
else:
    W = None

W=comm.scatter(W,root=0)

Q,R = TSQR(W,comm)

q = comm.gather(Q,root=0)

if rank==0:
    q = np.reshape(q,(m,n))
    print(np.linalg.cond(q)) # Should be 1
    print(np.linalg.norm(np.eye(n,n) - q.T @ q)) # Should be pretty much 0
