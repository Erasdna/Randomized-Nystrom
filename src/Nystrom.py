# Parallel randomized Nyström. Potentially two different implementations

from mpi4py import MPI
import numpy as np
import scipy
from TSQR import TSQR
from Sketching.Sketch import Sketch

def Nystrom(matrix : any, sketch : Sketch,n : int ,l : int ,k : int,seed : int,comm : any):
    """Parallel implementation of the Nyström approximation of a matrix

    Args:
        matrix (any): Matrix as built in the data.py file 
        sketch (Sketch): Sketching matrix
        n (int): Matrix size
        l (int): Oversampling parameter
        k (int): Approximation rank
        seed (int): Seed
        comm (any): MPI communicator

    Returns:
        _type_: _description_
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #First check that we can take sqrt of size:
    size_sqrt = int(np.sqrt(size))
    assert size_sqrt**2 == size
    assert n%size_sqrt==0 # check that n is divisble by sqrt size

    Omega = sketch(l,n//size_sqrt,seed)

    #Split communicator by rows and columns
    row_color = rank//size_sqrt
    row_comm = comm.Split(color = row_color,key=rank%size_sqrt)
    col_color = rank%size_sqrt
    if row_color==col_color:
        diag_color=1
    else:
        diag_color=0
    diag_comm = comm.Split(color = diag_color, key=row_color)
    
    # Get matrix
    A = matrix(n//size_sqrt,row_color,col_color)  

    #Initialize sketch. Each column generates the random matrix from
    #the same seed and therefore get the same Omega
    Omega.initialize_matrix(col_color)

    # C = A @ Omega
    C = Omega.apply_right(A)
    #Reduce along rows and collect in diagonal element
    C = row_comm.reduce(C,MPI.SUM,root=row_color)
    
    mu = 2.2e-16
    if row_color==col_color: 

        # Calculate Omega^T @ C with stabilizing element
        fac = diag_comm.allreduce(np.linalg.norm(C,'fro')**2,MPI.SUM)
        nu = mu*np.sqrt(fac)
        tmp = Omega.apply_right(np.eye(n//size_sqrt,n//size_sqrt)*nu)
        B = Omega.apply_left(C + tmp)
        B = diag_comm.allreduce(B,MPI.SUM)
        #Impose symmetry
        D=(B+B.T)/2

        #LAPACK cholesky
        L,_=scipy.linalg.lapack.dpotrf(D,lower=True,overwrite_a=True,clean=True)
        Z = scipy.linalg.solve_triangular(L,C.T,lower=True).T
        Q,R = TSQR(Z,diag_comm) # TSQR acts with assumption that Z is already scattered
        U,sigma,_ = np.linalg.svd(R) 
        Uhat = Q @ U[:,:k] 
        sig = sigma[:k]**2 - nu #Remove stabilizing nu
    else:
        Uhat = None
        sig = None 
    
    return Uhat,sig,A