# Parallel randomized Nyström. Potentially two different implementations

from mpi4py import MPI
import numpy as np
import scipy
from TSQR import TSQR

def Nystrom(matrix,sketch,n,l,k,seed,comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    #First check that we can take sqrt of size:
    size_sqrt = int(np.sqrt(size))
    assert size_sqrt**2 == size
    assert n%size_sqrt==0 # check that n is divisble by sqrt size

    #Split communicator by rows and columns
    row_color = rank//size_sqrt
    row_comm = comm.Split(color = row_color,key=rank%size_sqrt)
    col_color = rank%size_sqrt
    col_comm = comm.Split(color = col_color,key=rank//size_sqrt)
    if row_color==col_color:
        diag_color=1
    else:
        diag_color=0
    diag_comm = comm.Split(color = diag_color, key=row_color)
    
    # Get matrix
    
    A = matrix(n//size_sqrt,row_color,col_color, 0.0) 

    assert np.all(A == A.T)
    #Calculate and broadcast the sketching matrix
    if col_comm.Get_rank()==0:
        Omega = sketch(l,n//size_sqrt,seed,seed + rank)
    else:
        Omega = None
    
    Omega = col_comm.bcast(Omega,root = 0)

    #Calculate C, Allreduce along rows. We collect on diagonal procs to avoid
    #communicating omega
    if A is not None:
        C= A @ Omega
    else:
        C = np.zeros_like(Omega)
    C = row_comm.reduce(C,MPI.SUM,root=row_color)
    
    # Calculate Omega^T @ C
    mu=2.2e-16
    if row_color==col_color:
        fac = diag_comm.allreduce(np.linalg.norm(C,'fro')**2,MPI.SUM)
        nu = mu*np.sqrt(fac)
        C = C + nu*Omega # C = C + mu*Omega
        B = Omega.T @ C 
        B = diag_comm.reduce(B,MPI.SUM,root=0)
    else:
        B=None 
    
    if rank==0:
        # print(B.shape)
        eigs =scipy.linalg.eigvals((B+B.T)/2) 
        print(eigs[eigs<=0])
        # L = scipy.linalg.cholesky((B+B.T)/2) #enforce symmetry
        D=(B+B.T)/2
        L,info=scipy.linalg.lapack.dpotrf(D,lower=True,overwrite_a=True,clean=True)
        print(L.shape)
        print(info)
    else:
        L = None
    
    if row_color==col_color:
        L = diag_comm.bcast(L,root=0)
        Z = np.linalg.solve(L,C.T).T
        Q,R = TSQR(Z,diag_comm) # TSQR acts with assumption that Z is already scattered
        U,sigma,_ = np.linalg.svd(R) #R is small, we do the svd on all procs instead of communicating
        Uhat = Q @ U[:,:k] #Truncate U
        sig = sigma[:k]**2 - nu
    else:
        Uhat = None
        sig = None 
    
    return Uhat,sig,A