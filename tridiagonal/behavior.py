import time

from solvers import gaussian_elimination, thomas_ge
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random

rng = np.random.default_rng()

def generate_tridiagonal(n):
    m = 10
    return np.diag(np.full(m, n)) + np.diag(np.full(m-1, n), 1) + np.diag(np.ones(m-1), -1), np.ones(m).T


def comp_time(f, a, y):
    t0 = time.perf_counter()
    _ = f(a, y)
    t1 = time.perf_counter()
    return t1 - t0

n_max = 1000
n_range = np.linspace(0, 1e-10, n_max)
n_try = 1

t_ge = np.empty(n_max)
t_th = np.empty(n_max)
for i in range(n_max):
    system = generate_tridiagonal(i+3)
    t_ge[i] = comp_time(np.linalg.solve, *system)
    t_th[i] = comp_time(thomas_ge, *system)


fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
fig.suptitle("Time-Cost Scaling for\nSolving Tridiagonal Matrices")

ax.scatter(n_range, t_ge, 0.1, 'r', label="Gaussian elimination")
ax.scatter(n_range, t_th, 0.1, 'b', label="Thomas algorithm")
ax.set(xlabel="$n$", ylabel="Time cost [s]", yscale='log')
ax.legend()

# plt.savefig("algo_behavior.png", dpi=300)
plt.show()
