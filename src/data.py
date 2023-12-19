# Parallel creation of example matrices of rank R, with param q and size n
import numpy as np
import scipy.sparse as sp

def poly_factory(q : float ,R : int):
    return lambda n,row,col: polydecay(n,q,R,row,col)

def polydecay(n : int,q : float ,R : int ,row : int ,col : int) -> np.ndarray:
    """Build polynomial decay data matrix 

    Args:
        n (int): size of the matrix
        q (float): decay factor
        R (int): effective rank
        row (int): row position in parallel processor grid
        col (int): column position in parallel processor grid

    Returns:
        (np.ndarray): Matrix belonging on processor (row,col)
    """
    if row!= col:
        return np.zeros((n,n))
    else:
        ind = np.arange(n*row,n*(row+1),1)
        d = np.where(ind<R, 1.0,(ind - R + 2.0)**(q))
        return np.diag(1/d)  

def exp_factory(q : float ,R : int):
    return lambda n,row,col: expdecay(n,q,R,row,col)

def expdecay(n : int ,q : float , R: int, row : int, col : int) -> np.ndarray:
    """Build exponential decay data matrix 

    Args:
        n (int): size of the matrix
        q (float): decay factor
        R (int): effective rank
        row (int): row position in parallel processor grid
        col (int): column position in parallel processor grid

    Returns:
        (np.ndarray): Matrix belonging on processor (row,col)
    """
    if row!= col:
        return np.zeros((n,n))
    else:
        ind = np.arange(n*row,n*(row+1),1)
        d = np.where(ind<R, 1.0,10.0**(-(ind - R + 1.0)*q))
        return np.diag(d) 