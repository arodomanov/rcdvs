import numpy as np
import scipy.sparse as ssp
from functions import *
from auxiliary import *

# Generate a random vector `u` uniformly distributed on the unit sphere in `R^n`
def random_uniform_direction(n):
    v = np.random.randn(n)
    u = v / np.linalg.norm(v)
    return u

# Generate a sparse random vector `u` (`n-by-1` CSC matrix) from the unit sphere in `R^n` such that
# `u` has exactly `p` non-zeros at random uniformly distributed positions; the non-zero part is
# uniformly distributed on the unit sphere in `R^p`
def sparse_random_uniform_direction(n, p):
    assert(p <= n)
    s = np.random.choice(n, p, replace=False)
    u_s = random_uniform_direction(p)
    return sparse_vector(u_s, s, n)


# Construct an `m x n` diagonal matrix with the vector `d` on the diagonal
def diagonal_matrix(m, n, d):
    # Construct a diagonal matrix with
    A = np.zeros((m, n))
    i, j = np.indices(A.shape)
    A[i == j] = d
    return A

# Generate a random `m x n` matrix with given singular values using a series of left/right
# Householder reflections in random directions uniformly distributed on the unit spheres in `R^m` and `R^n`
def random_matrix(m, n, singvals, n_reflections=10):
    assert(np.all(singvals >= 0))
    A = diagonal_matrix(m, n, singvals)
    for i_reflection in range(n_reflections):
        u = random_uniform_direction(n)
        v = random_uniform_direction(m)
        A = A - 2 * A.dot(np.outer(u, u))
        A = A - 2 * np.outer(v, v).dot(A)
    return A

# Generate a random `n x n` symmetric matrix with given eigenvalues using a series of symmetric
# Householder reflections in random directions uniformly distributed on the unit sphere in `R^n`
def random_symmetric_matrix(eigvals, n_reflections=10):
    assert(np.all(eigvals >= 0))
    n = eigvals.size
    A = np.diag(eigvals)
    for i in range(n_reflections):
        u = random_uniform_direction(n)
        uuT = np.outer(u, u)
        A = A - 2 * A.dot(uuT)
        A = A - 2 * uuT.dot(A)
    return A

# `p` is the sparsity level (see `sparse_random_uniform_direction`)
def sparse_random_matrix(m, n, p, singvals, n_reflections=10):
    assert(np.all(singvals >= 0))
    A = ssp.diags(singvals, shape=(m, n)).tocsc()
    for i_reflection in range(n_reflections):
        u = sparse_random_uniform_direction(n, p)
        v = sparse_random_uniform_direction(m, p)
        A = A - 2 * A * u * u.T
        A = A - 2 * v * v.T * A
    A.sort_indices()
    assert(ssp.isspmatrix_csc(A))
    assert(A.has_sorted_indices)
    return A

def generate_random_quadratic_instance(n, eigval1, eigval2):
    assert(n >= 2)
    eigvals = np.concatenate(([eigval1], [eigval2], np.ones(n - 2)))
    A = random_symmetric_matrix(eigvals)
    x_bar = np.random.rand(n) * 2 - 1
    b = A.dot(x_bar)
    f_opt = -0.5 * np.dot(b, x_bar)
    return A, b, f_opt

def generate_random_separable_instance(m, n, eigval1, eigval2, mu, function):
    q = min(m, n)
    assert(q >= 2)
    gamma = 0  # No regularization by default
    if function == FUNCTION_LOGISTIC: gamma = 1
    L = get_smoothness_constant(function, mu)
    singvals = (np.concatenate(([eigval1], [eigval2], np.ones(q - 2))) / L)**0.5
    A = random_matrix(m, n, singvals)
    if function == FUNCTION_LEAST_SQUARES or function == FUNCTION_HUBER:
        x_bar = np.random.rand(n) * 2 - 1
        b = A.dot(x_bar)
        f_opt = 0
    if function == FUNCTION_LOGISTIC:
        b = np.sign(np.random.randn(m))
        f_opt = compute_sepfunc_f_opt(A, b, gamma, mu, function)
    return A, b, gamma, f_opt

# `p` is the sparsity level (see `sparse_random_matrix`)
def generate_random_sparse_separable_instance(m, n, p, eigval1, eigval2, mu, function):
    q = min(m, n)
    assert(q >= 2)
    gamma = 0  # No regularization by default
    if function == FUNCTION_LOGISTIC: gamma = 1
    L = get_smoothness_constant(function, mu)
    singvals = (np.concatenate(([eigval1], [eigval2], np.ones(q - 2))) / L)**0.5
    A = sparse_random_matrix(m, n, p, singvals)
    if function == FUNCTION_LEAST_SQUARES or function == FUNCTION_HUBER:
        x_bar = np.random.rand(n) * 2 - 1
        b = A.dot(x_bar)
        f_opt = 0
    if function == FUNCTION_LOGISTIC:
        b = np.sign(np.random.randn(m))
        f_opt = compute_sepfunc_f_opt(A, b, gamma, mu, function)
    return A, b, gamma, f_opt
