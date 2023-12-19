from mpi4py import MPI
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time 

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory, exp_factory
from Sketching.LASO import LASO
from Sketching.SSO import SSO
from Sketching.Gaussian import Gaussian
from Sketching.SASO import SASO
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

    k=20
    n=int(sys.argv[2])
    
    diag_size = int(np.sqrt(size))
    mult = [2,5,10,15]

    sketch = [Gaussian,SASO,LASO,SSO]

    d1 = len(mult)
    d2 = k
 
    err = np.zeros((len(sketch),d1,d2))
    err2 = np.zeros((len(sketch),d1,d2))
    timing = np.zeros((len(sketch),d1,d2))
    tot = np.zeros((len(sketch),d1,d2))

    for j in range(k):
        for jj,m in enumerate(mult):
            for nn,s in enumerate(sketch):
                tot[nn,jj,j]=j+9
                matrix = factory(q=q,R=10)
                start = time.perf_counter()
                U,sigma,A = Nystrom(
                    matrix=matrix,
                    sketch=s,
                    n=n,
                    l=m*(j+9),
                    k=j+9,
                    seed=55,
                    comm=comm
                )
                print("Time: ", time.perf_counter()-start)
                if rank==0:
                    timing[nn,jj,j] = time.perf_counter()-start
                    Nys = U @ np.diag(sigma) @ U.T
                    uu,ss,vv = np.linalg.svd(A)
                    err[nn,jj,j] = np.linalg.norm(A - Nys, 'nuc')/np.linalg.norm(A , 'nuc')
                    err2[nn,jj,j] = (np.linalg.norm(A - Nys, 'nuc')/np.linalg.norm(A - uu[:,:j+6] @ np.diag(ss[:j+6]) @ vv[:j+6,:], 'nuc'))-1
                    #print(j, " ", err[nn,jj,j])
                    #print(err2[nn,jj,j])
                comm.Barrier()

    names = ["Gaussian", "SASO(8)", "LASO(60)","SSO(16)"]
    markers = ["^","<", ">", "o"]
    filename = os.getcwd() + "/Figures/" + sys.argv[1] + "/" + sys.argv[2] + "/rank/"
    
    for i,n in enumerate(names):
        fig,ax = plt.subplots()
        for j,d in enumerate(mult):
            ax.plot(tot[i,j,:],err[i,j,:],label="l=" + str(d) + "k",marker=markers[j])
        ax.set_xlabel("Rank")
        ax.set_ylabel("$||A - [A_{Nyst}]_k||_*/||A||_*$")
        ax.legend()
        ax.grid()
        fig.savefig(filename + n + "_A_error_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".eps",bbox_inches='tight')
        fig.savefig(filename + n + "_A_error_" + sys.argv[1] + "n=" + sys.argv[2] +"_q=" + sys.argv[3] + ".png",bbox_inches='tight')

    for i,n in enumerate(names):
        fig,ax = plt.subplots()
        for j,d in enumerate(mult):
            ax.plot(tot[i,j,:],err2[i,j,:],label="l=" + str(d) + "k",marker=markers[j])
        ax.set_xlabel("Rank")
        ax.set_ylabel("$||A - [A_{Nyst}]_k||_*/||A - [A_{SVD}]_k||_* - 1$")
        ax.legend()
        ax.grid()
        fig.savefig(filename + n + "_SVD_error_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".eps",bbox_inches='tight')
        fig.savefig(filename + n + "_SVD_error_" + sys.argv[1] + "n=" + sys.argv[2] +"_q=" + sys.argv[3] + ".png",bbox_inches='tight')
