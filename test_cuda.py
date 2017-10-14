import math
from numba import vectorize, cuda
import numpy as np
from timeit import default_timer as timer

@vectorize(['float32(float32, float32)'], target='cuda')
def cuVectorAdd(a, b):
    return a + b

def VectorAdd(a, b):
    return np.sum([a, b], axis=0)


N = 32000000
dtype = np.float32

# prepare the input
A = np.ones(N, dtype=dtype)
B = np.ones(N, dtype=dtype)

start = timer()
C = VectorAdd(A, B)
vactoradd_time = timer() - start 
print("Tempo %f segundos "%vactoradd_time)  # print result
print(C[0:10])

start = timer()
C = cuVectorAdd(A, B)
vactoradd_time = timer() - start 
print("Tempo %f segundos "%vactoradd_time)  # print result
print(C[0:10])


import cudamat as cm

n, p = int(2e3), int(40e3)
A = np.random.randn(n, p)
B = np.random.randn(p, n)
A @ B

cm.cublas_init()
cm.CUDAMatrix.init_random()
A_cm = cm.empty((n, p)).fill_with_randn()
B_cm = cm.empty((p, n)).fill_with_randn()
A_cm.dot(B_cm)
cm.cublas_shutdown()