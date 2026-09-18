import numpy as np, time

n = 4000
a = np.random.rand(n, n)
b = np.random.rand(n, n)
a @ b  # warm-up run

start = time.perf_counter()
a @ b
elapsed = time.perf_counter() - start

gflops = 2 * n**3 / elapsed / 1e9  # matrix multiply = 2·n³ operations
print(f"{elapsed:.2f} s  →  {gflops:.0f} GFLOPS")