from mpi4py import MPI
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time 

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory, exp_factory
from Sketching import SRHT,Gaussian, SASO
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
    
    diag_size = int(np.sqrt(size))
    its=20

    sketch = [Gaussian,SRHT, SASO]
    err = np.zeros((len(sketch),its))
    err2 = np.zeros((len(sketch),its))
    timing = np.zeros((len(sketch),its))

    tot = np.zeros((len(sketch),its))

    for j in range(its):
        for nn,s in enumerate(sketch):
            l = k*(j+2)
            tot[nn,j]=l
            matrix = factory(q=q,R=10)
            start = time.perf_counter()
            U,sigma,A = Nystrom(
                matrix=matrix,
                sketch=s,
                n=n,
                l=l,
                k=k,
                seed=55,
                comm=comm
            )
            print("Time: ", time.perf_counter()-start)
            if rank==0:
                timing[nn,j] = time.perf_counter()-start
            comm.Barrier()

    filename = os.getcwd() + "/Figures/oversampling/"
    
    fig3,ax3 = plt.subplots()
    ax3.plot(tot[0,:],timing[0,:],label="Gaussian",marker="^")
    ax3.plot(tot[1,:],timing[1,:],label="SRHT", marker="<")
    ax3.plot(tot[2,:],timing[2,:],label="SASO(8)", marker=">")
    ax3.set_xlabel("Oversampling parameter (rank=10)")
    ax3.set_ylabel("Runtime [s]")
    ax3.legend()
    ax3.grid()
    fig3.savefig(filename + "timing_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".eps",bbox_inches='tight')
    fig3.savefig(filename + "timing_" + sys.argv[1] + "n=" + sys.argv[2] + "_q=" + sys.argv[3] + ".png",bbox_inches='tight')