"""Package `core`. Environment settings to apply before any import of torch/faiss.

On macOS, faiss-cpu and PyTorch each ship their own OpenMP runtime (libomp). Loading the
second one aborts the process ("OMP: Error #15"); KMP_DUPLICATE_LIB_OK=TRUE lets it
continue, but crashes (SIGSEGV in __kmp_suspend) occur as soon as an OpenMP worker is
started (observed on 28/10 with CoverHunter). OMP_NUM_THREADS=1 removes the workers:
heavy computations run on MPS, and FAISS over 25k vectors remains instantaneous.
These values are defaults: the environment can override them."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
