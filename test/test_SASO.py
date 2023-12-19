from mpi4py import MPI
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Sketching.LASO import LASO

mat = np.random.randn(1000,1000)
sketch = LASO(50,1000,3,d=8)
sketch.initialize_matrix(0)
sketch.apply_right(mat)