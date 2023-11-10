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
    A = matrix(n//size_sqrt,row_color,col_color) # Or something similar?

    #Calculate and broadcast the sketching matrix
    if col_comm.Get_rank()==0:
        Omega = sketch(l,n//size_sqrt,seed,seed + rank)
    else:
        Omega = None
    Omega = col_comm.bcast(Omega,root = 0)
    
    #Calculate C, Allreduce along rows. We collect on diagonal procs to avoid
    #communicating omega
    C= A @ Omega
    C = row_comm.reduce(C,MPI.SUM,root=row_color)

    # Calculate Omega^T @ C
    if row_color==col_color:
        B = Omega.T @ C 
        B = diag_comm.reduce(B,MPI.SUM,root=0)
    else:
        B=None 
    
    if rank==0:
        #DEBUG: Check that implementation works for B psd
        B = B + 2*np.eye(B.shape[0],B.shape[1])
        L = np.linalg.cholesky(B)
    else:
        L = None
    
    if row_color==col_color:
        L = diag_comm.bcast(L,root=0)
        Z = scipy.linalg.solve_triangular(L,C.T).T
        Q,R = TSQR(Z,diag_comm) # TSQR acts with assumption that Z is already scattered
        U,sigma,VT = np.linalg.svd(R) #R is small, we do the svd on all procs instead of communicating
        Uhat = Q @ U[:,:k] #Truncate U
        sigmaU = np.diag(sigma[:k]**2) @ Uhat.T
        Us = diag_comm.allgather(sigmaU)
        sigmaU_full = np.reshape(Us, (sigmaU.shape[0],len(Us)*sigmaU.shape[1]))
        Nyst = Uhat @ sigmaU_full
        Nyst_collect = diag_comm.gather(Nyst,root=0)
        if rank==0:
            Nyst = np.reshape(Nyst_collect, (len(Nyst_collect)*Nyst.shape[0],Nyst.shape[1]))
    else:
        Nyst = None
    
    return Nyst