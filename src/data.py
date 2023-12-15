#Function for generating matrices

# Parallel creation of example matrices of rank R, with param q and size n
import numpy as np
import scipy.sparse as sp

def poly_factory(q,R):
    return lambda n,row,col: polydecay(n,q,R,row,col)

def polydecay(n,q,R,row,col):
    if row!= col:
        return np.zeros((n,n)) #sp.csr_matrix((n,n),dtype=float)
    else:
        ind = np.arange(n*row,n*(row+1),1)
        d = np.where(ind<R, 1.0,(ind - R + 2.0)**(q))
        return np.diag(1/d) #sp.diags(1/d) 

def exp_factory(q,R):
    return lambda n,row,col: expdecay(n,q,R,row,col)

def expdecay(n,q,R,row,col):
    if row!= col:
        return np.zeros((n,n))#sp.csr_matrix((n,n),dtype=float)
    else:
        ind = np.arange(n*row,n*(row+1),1)
        d = np.where(ind<R, 1.0,10.0**(-(ind - R + 1.0)*q))
        return np.diag(d) #sp.diags(d) 