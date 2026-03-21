import scipy.sparse, scipy.sparse.linalg, numpy as np, time

# Simulate a Laplacian-like matrix (banded, structured sparsity)
N = 10000
diags = [np.ones(N)*10, -np.ones(N-1), -np.ones(N-1), -np.ones(N-5), -np.ones(N-5)]
offsets = [0, 1, -1, 5, -5]
A = scipy.sparse.diags(diags, offsets, shape=(N, N), format='csr')
b = np.random.rand(N)

# scipy splu
factor = scipy.sparse.linalg.splu(A.tocsc())
times = []
for _ in range(50):
    t0 = time.perf_counter()
    factor.solve(b)
    times.append(time.perf_counter() - t0)
print(f"scipy splu forward-solve (structured): {np.median(times)*1000:.2f} ms")

# pypardiso
import pypardiso
solver = pypardiso.PyPardisoSolver()
solver.factorize(A)
times = []
for _ in range(50):
    t0 = time.perf_counter()
    solver.solve(A, b)
    times.append(time.perf_counter() - t0)
print(f"pypardiso forward-solve (structured): {np.median(times)*1000:.2f} ms")