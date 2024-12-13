import numpy as np
from scipy.linalg import lu, solve


def gaussian_elimination(t: np.ndarray, y: np.ndarray):
    return np.linalg.solve(t, y)


def thomas_ge(t: np.ndarray, y: np.ndarray):
    n = t.shape[0]
    cp = np.empty(n-1)
    yp = np.empty(n)
    x = np.empty(n)

    # Back sub
    yp[0] = y[0] / t[0, 0]
    cp[0] = t[0, 1] / t[0, 0]
    for i in range(1, n):
        if i < n-1:
            cp[i] = t[i, i+1] / (t[i, i] - t[i, i-1] * cp[i-1])
        yp[i] = (y[i] - t[i, i-1] * yp[i-1]) / (t[i, i] - t[i, i-1] * cp[i-1])
    yp[n-1] =  (y[n-1] - t[n-1, n-2] * yp[n-2]) / (t[n-1, n-1] - t[n-1, n-2] * cp[n-2])

    # Forward sub
    x[n-1] = yp[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = yp[i-1] - cp[i-1] * x[i]
    return x


def thomas_lu(t: np.ndarray, y: np.ndarray):
    # pl, u = lu(t, permute_l=True)
    n = t.shape[0]
    l = np.empty(n-1)
    u = np.empty(n)
    rho = np.empty(n)
    x = np.empty(n)

    u[0] = rho[0] = t[0, 0]
    for j in range(1, n):
        # LU factorization
        l[j-1] = t[j, j-1] / u[j-1]
        u[j] = t[j, j] - l[j-1] * t[j-1, j]

        # Forward sub
        rho[j] = y[j] - l[j-1] * y[j-1]

    # Back sub
    x[n-1] = rho[n-1] / u[n-1]
    for k in range(n-2, 0, -1):
        x[k] = (rho[k] - t[k-1, k] * x[k+1]) / u[k]

    return x


a = np.array([
    [5, 0, 0],
    [6, 1, 1.001],
    [0, -9, -9]
])
r = np.array([
    6, 1, 1
]).T

# print(thomas_lu(a, r))
print(thomas_ge(a, r))
print(solve(a,r))