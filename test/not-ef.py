import numpy as np
import time
import random

N = 10_000_000
A = np.arange(N, dtype=np.int64)
# ---------------------------

# Sequential Access (GOOD)

# ---------------------------

start = time.time()

s = 0
for i in range(N):
    s += A[i]

end = time.time()
print("Sequential access time:", end - start)

# ---------------------------

# Random Access (BAD)

# ---------------------------

indices = np.random.randint(0, N, size=N)

start = time.time()

s = 0
for i in indices:
    s += A[i]

end = time.time()
print("Random access time:", end - start)
