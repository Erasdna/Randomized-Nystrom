from mpi4py import MPI
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Nystrom import Nystrom
from data import poly_factory
from Sketching import SRHT

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matrix = poly_factory(q=2.0,R=10)
Nys = Nystrom(
    matrix=matrix,
    sketch=SRHT,
    n=1024,
    l=50,
    k=10,
    seed=55,
    comm=comm
)
if rank==0:
    print(Nys.shape)

