import IMP
import IMP.bff
import IMP.pmi
from mpi4py import MPI
import numba.cuda
import smlm_score
import sys

print("-" * 30)
print(f"Python:   {sys.version.split()[0]}")
print(f"IMP:      {IMP.__version__}")
print(f"MPI Size: {MPI.COMM_WORLD.Get_size()}")
print(f"CUDA:     {numba.cuda.is_available()}")
print(f"Project:  {smlm_score.__file__}")
print("-" * 30)
