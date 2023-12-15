import os
import sys
import numpy as np
from scipy.linalg import hadamard
import time

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

#from Sketching import fht,fht2,SRHT
from Sketching.SRHT import SRHT

size = 2**(13)
rng = np.random.default_rng(seed=5)
rad = rng.choice([-1,1],size)
test_matrix = np.random.randn(size,size)
A = np.eye(size,size) #np.random.randn(size,size)
ss = SRHT(50,size,3)

# start=time.perf_counter()
# mat = SRHT(A,50,size,3,3)
# res_pytorch=mat
# print("Scipy matrix: ", time.perf_counter()- start)

ss.initialize_matrix(0)
start2 = time.perf_counter()
res_numpy = ss.apply_right(A)
print("Class: ", time.perf_counter()-start2)
