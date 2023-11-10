from mpi4py import MPI
import numpy as np
import mpi4py

def TSQR(W,comm):
    """TSQR parallel algorithm for calculating QR decomposition of a matrix using Householder rotations

    Args:
        W (Matrix): Matrix W from which QR is calculated
        comm (): MPI communicator

    Returns:
        Matrix,Matrix: Q,R such that W=QR
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    #We assume that W has already been scattered on the communicator
    W_size = W.shape
    
    n = np.ceil(np.log2(size)).astype(int)
    
    active = [np.arange(size)]
    color=0
    Qq = []
    R_comm = [comm.Split(color=size+1,key=rank)]
    #We move from leaves to root in the binary graph to find R
    for i in range(n):
        if np.any(active[i]==rank):
            R_rank = R_comm[-1].Get_rank()
            R_size = R_comm[-1].Get_size()
            Q,R = np.linalg.qr(W)
            if R_rank%2==1:
                R_comm[-1].Send(R,dest = R_rank-1)
                Qq.append(Q)
                color=0
            elif R_rank%2==0 and (R_rank != R_size-1):
                R_collect= np.empty_like(R)
                Qq.append(Q)
                R_comm[-1].Recv(R_collect,source = R_rank+1)
                W = np.vstack([R, R_collect])
                color=1
            else:
                color=1 # If nb of procs is odd
        R_comm.append(comm.Split(color=color,key = R_rank//2))
        active.append(active[-1][::2]) # Reduce by half the number of active processes
        
    active.pop(-1)
    R_comm.pop(-1)
    if rank==0:
        Q,R = np.linalg.qr(W) # Top level

    #We then move back down the graph to collect Q
    for i in range(n-1,-1,-1):
        if np.any(active[i]==rank):
            Q_rank = R_comm[-1].Get_rank()
            Q_size = R_comm[-1].Get_size()
            if Q_rank%2==0 and (Q_rank != Q_size-1):
                R_comm[-1].Send(Q[Q.shape[0]//2:,:],dest = Q_rank + 1) 
                Q_l = Qq.pop(-1)
                Q = Q_l @ Q[:Q_l.shape[-1],:]
            elif Q_rank%2==1:
                Q = np.empty_like(R)
                R_comm[-1].Recv(Q,source =Q_rank - 1)
                Q_l = Qq.pop(-1)
                Q = Q_l @ Q[-Q_l.shape[-1]:,:]
        R_comm.pop(-1)
    
    R = comm.bcast(R,root=0)
    return Q,R
