import numpy as np
from time import time

start_time = time()
A = np.arange(1000000)
s = 0
for i in range(len(A)):
    s += A[i]

print("Time taken: ", time() - start_time)