---
header-includes:
  - \usepackage{algorithm}
  - \usepackage{algpseudocode}
---

# Randomized-Nystrom
EPFL MATH 505 project on Randomized Nyström 

We implement a parallel implementation of the Randomized Nyström approximation following [this paper](https://arxiv.org/abs/1706.05736). The parallel implementation is based on [MPI4py](https://mpi4py.readthedocs.io/en/stable/)    

The Nyström method is implemented in `src/Nystrom.py`

We implement four different sketching matrices:
- Gaussian: `src/Sketching/Gaussian.py`
- Short-Axis-Sparse: `src/Sketching/SASO.py`
- Long-Axis-Sparse: `src/Sketching/SSO.py`
- Sparse Sketching Operator: `src/Sketching/SSO.py` 