#Function for generating matrices

# Parallel creation of example matrices of rank R, with param q and size n
import numpy as np
import scipy.sparse as sp

def poly_factory(q,R):
    return lambda n,row,col: polydecay(n,q,R,row,col)

def polydecay(n,q,R,row,col):
    if row!= col:
        return sp.csr_matrix((n,n))
    else:
        ind = np.arange(n*row,n*(row+1),1)
        d = np.where(ind<R, 1,(ind - R +1)**(-q))
        return sp.diags(d) 