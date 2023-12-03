from mpi4py import MPI
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Sketching import SASO,SRHT,Gaussian

start_SASO = time.perf_counter()
out = SASO(20,4096,55,0,fac=0.2)
print(time.perf_counter() - start_SASO)

start_SRHT = time.perf_counter()
SRHT(20,4096,55,0)
print(time.perf_counter() - start_SRHT)

start_Gaussian = time.perf_counter()
Gaussian(20,4096,55,0)
print(time.perf_counter() - start_Gaussian)