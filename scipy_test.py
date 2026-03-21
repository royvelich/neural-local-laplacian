import scipy.sparse, scipy.sparse.linalg, numpy as np, time

A = scipy.sparse.random(10000, 10000, density=0.001, format='csr')
A = A + A.T + 10 * scipy.sparse.eye(10000)
b = np.random.rand(10000)

# scipy splu
factor = scipy.sparse.linalg.splu(A.tocsc())
times = []
for _ in range(20):
    t0 = time.perf_counter()
    factor.solve(b)
    times.append(time.perf_counter() - t0)
print(f"scipy splu forward-solve: {np.median(times)*1000:.2f} ms")