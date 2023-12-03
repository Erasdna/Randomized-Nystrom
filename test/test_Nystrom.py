from mpi4py import MPI
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory
from Sketching import SRHT,Gaussian,SASO

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
diag_size = int(np.sqrt(size))
k=10
l=int(20*k)
n=(1024)

matrix = poly_factory(q=1.0,R=10)
U,sigma,A = Nystrom(
    matrix=matrix,
    sketch=SASO,
    n=n,
    l=l,
    k=k,
    seed=55,
    comm=comm
)

U = comm.gather(U,root=0)
A = comm.gather(A,root=0)
if rank==0:
    diag_A = np.zeros(n)
    u = np.zeros((n,k))
    for ind in range(len(A)):
        if ind%diag_size==ind//diag_size:
            i=ind%diag_size
            diag_A[i*(n//diag_size):(i+1)*(n//diag_size)] = A[ind].diagonal()
            u[i*(n//diag_size):(i+1)*(n//diag_size),:] = U[ind]
    Nys = u @ np.diag(sigma) @ u.T
    print(np.linalg.norm(np.diag(diag_A) - Nys, 'nuc')/np.linalg.norm(np.diag(diag_A) , 'nuc'))

