from mpi4py import MPI
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from data import polydecay

n = 100
q=2.0
R=10
row=0
col=0

polydecay(n,q,R,row,col)
