from mpi4py import MPI
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time 

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory, exp_factory
from Sketching.SASO import SASO
from Sketching.LASO import LASO
from Sketching.SSO import SSO
from cycler import cycler

if __name__=="__main__":
    plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:blue', 'tab:red', 'tab:green', 'k', 'tab:purple']) +
                                cycler('linestyle', ['-', '-.', '--', ':','-'])))

    comm=MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size==1

    if sys.argv[1]=="poly":
        factory = poly_factory
    elif sys.argv[1]=="exp":
        factory = exp_factory
    
    q = float(sys.argv[3]) 

    k=10
    n=int(sys.argv[2])

    if sys.argv[4]=="SASO":
        so = SASO
        sketch = [8,12,16,20]
    elif sys.argv[4]=="LASO":
        so = LASO
        sketch = [12,30,60,120,300]
    elif sys.argv[4]=="SSO":
        so = LASO
        sketch = [8,12,16,20]

    
    diag_size = int(np.sqrt(size))
    its=20

    
    err = np.zeros((len(sketch),its))
    err2 = np.zeros((len(sketch),its))
    timing = np.zeros((len(sketch),its))

    tot = np.zeros((len(sketch),its))

    for j in range(its):
        for nn,s in enumerate(sketch):
            l = k*(j+3)
            print(l)
            tot[nn,j]=l
            matrix = factory(q=q,R=10)
            mat = lambda l,dim,seed: so(l,dim,seed,s)
            start = time.perf_counter()
            U,sigma,A = Nystrom(
                matrix=matrix,
                sketch=mat,
                n=n,
                l=l,
                k=k,
                seed=55,
                comm=comm
            )
            print("Time: ", time.perf_counter()-start)
            if rank==0:
                timing[nn,j] = time.perf_counter()-start
                Nys = U @ np.diag(sigma) @ U.T
                uu,ss,vv = np.linalg.svd(A)
                err [nn,j] = np.linalg.norm(A - Nys, 'nuc')/np.linalg.norm(A , 'nuc')
                err2[nn,j] = (np.linalg.norm(A - Nys, 'nuc')/np.linalg.norm(A - uu[:,:k] @ np.diag(ss[:k]) @ vv[:k,:], 'nuc'))-1
                print(j, " ", err[nn,j])
                print(err2[nn,j])
            comm.Barrier()

    filename = os.getcwd() + "/Figures/SO/" + sys.argv[4] 
    fig,ax = plt.subplots()
    for i,el in enumerate(err):
        ax.plot(tot[i,:],el,label=sys.argv[4] + "(" + str(sketch[i]) + ")")
    ax.set_xlabel("Oversampling parameter (rank=10)")
    ax.set_ylabel("$||A - [A_{Nyst}]_k||_*/||A||_*$")
    ax.legend()
    ax.grid()
    fig.savefig(filename + "_A_error_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".eps",bbox_inches='tight')
    fig.savefig(filename + "_A_error_" + sys.argv[1] + "n=" + sys.argv[2] +"_q=" + sys.argv[3] + ".png",bbox_inches='tight')

    fig2,ax2 = plt.subplots()
    for i,el in enumerate(err2):
        ax2.plot(tot[i,:],el,label= sys.argv[4] + "(" + str(sketch[i]) + ")")
    ax2.set_xlabel("Oversampling parameter (rank=10)")
    ax2.set_ylabel("$||A - [A_{Nyst}]_k||_*/||A - [A_{SVD}]_k||_* - 1$")
    ax2.legend()
    ax2.grid()
    fig2.savefig(filename + "_SVD_error_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".eps",bbox_inches='tight')
    fig2.savefig(filename + "_SVD_error_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".png",bbox_inches='tight')
