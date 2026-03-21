import pypardiso, scipy.sparse, numpy as np, time

A = scipy.sparse.random(10000, 10000, density=0.001, format='csr')
A = A + A.T + 10 * scipy.sparse.eye(10000)
b = np.random.rand(10000)

solver = pypardiso.PyPardisoSolver()
solver.factorize(A)

# Time just the forward-solve
times = []
for _ in range(20):
    t0 = time.perf_counter()
    solver.solve(A, b)
    times.append(time.perf_counter() - t0)

print(f"Forward-solve: {np.median(times)*1000:.2f} ms")