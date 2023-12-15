from mpi4py import MPI
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory
from Sketching import SRHT,Gaussian,SASO

if __name__=="__main__":

    comm=MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    k=10
    diag_size = int(np.sqrt(size))
    l=5*k

    N = int(sys.argv[3])
    if sys.argv[2]=="Weak":
        n=diag_size*N
    elif sys.argv[2]=="Strong":
        n=N
    

    if sys.argv[1] == "Gaussian":
        sketch = Gaussian
    elif sys.argv[1]=="SRHT":
        sketch = SRHT 
    elif sys.argv[1]=="SASO":
        sketch=SASO
    filename = os.getcwd() + "/Figures/Timing/" + sys.argv[1] + "/" + sys.argv[2] + "_n=" + sys.argv[3] + "_"

    matrix = poly_factory(q=1.0,R=10)

    start = time.perf_counter()
    U,sigma,A = Nystrom(
                matrix=matrix,
                sketch=sketch,
                n=n,
                l=l,
                k=k,
                seed=55,
                comm=comm
            )
    #comm.Barrier()
    if rank==0:
        t = time.perf_counter()-start
        print("Procs: ", size, " time: ", t)
        with open(filename+"data.txt", "a") as myfile:
            myfile.write(str(size) + " " + str(t) + "\n")
