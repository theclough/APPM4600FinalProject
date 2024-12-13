import time
from solvers import gaussian_elimination, thomas_ge
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random

rng = np.random.default_rng()


def generate_tridiagonal(n):
    return np.diag(np.full(n, 2)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1), np.ones(n).T


def comp_time(f, a, y):
    t0 = time.perf_counter()
    _ = f(a, y)
    t1 = time.perf_counter()
    return t1 - t0


n_max = 100
n_range = np.arange(3, n_max+3, 1)
n_try = 100

t_ge = np.empty(n_max)
t_th = np.empty(n_max)
for i in range(n_max):
    system = generate_tridiagonal(i+3)
    t_temp_ge = np.empty(n_try)
    t_temp_th = np.empty(n_try)
    for j in range(n_try):
        t_temp_ge[j] = comp_time(np.linalg.solve, *system)
        t_temp_th[j] = comp_time(thomas_ge, *system)
    t_ge[i] = np.min(t_temp_ge)
    t_th[i] = np.min(t_temp_th)


fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
fig.suptitle("Time-Cost Scaling for\nSolving Tridiagonal Matrices")

ax.scatter(n_range, t_ge, 0.1, 'r', label="Gaussian elimination")
ax.scatter(n_range, t_th, 0.1, 'b', label="Thomas algorithm")
ax.set(xscale='log', yscale='log', xlabel="$n$", ylabel="Time cost [s]")
ax.legend()

plt.savefig("algo_performance.png", dpi=300)
# plt.show()
