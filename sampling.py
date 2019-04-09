import numpy as np
from numba import jit
from auxiliary import *

##########################################################################
# Define global constants
SAMPLING_UNIFORM = 1
SAMPLING_VOLUME = 2
##########################################################################

##########################################################################
########################### Uniform sampling #############################
##########################################################################
# Sample uniform integer in [0 .. n-1]
@jit('i4(i4)', nopython=True, cache=True)
def sample_uniform(n):
    u = np.random.rand()
    return int(n * u)

# Uniform sampling from `[n] \choose \tau`
# `n` is assumed to be large, and `tau` is assumed to be small (e.g. `tau <= 4`)
# This is not the most efficient version, but it is fine for such small values of `tau`.
@jit('i4[:](i4, i4)', nopython=True, cache=True)
def us_sample(n, tau):
    s = np.zeros(tau, dtype=np.int32)
    for i in range(tau):
        k = sample_uniform(n - i)
        for j in range(i):
            if k >= s[j]:
                k += 1
        s[i] = k
        s[:i + 1] = np.sort(s[:i + 1])
    return s

##########################################################################
########################### Volume sampling ##############################
##########################################################################

# Preprocessing procedure for `tau`-element volume sampling with respect to `B`
@jit('Tuple((f8[:], i4[:]))(f8[:,:], i4)', nopython=True, cache=True)
def vs_preprocess(B, tau):
    n = B.shape[0]
    num = binom(n, tau)
    proba_cumsum = np.zeros(num)
    idx = np.zeros(num * tau, dtype=np.int32)
    cumsum = 0
    r = 0
    s = np.zeros(tau, dtype=np.int32)
    for i in range(tau):
        s[i] = i
    while s.size > 0:
        B_ss = get_principal_submatrix(B, s)
        cumsum += np.linalg.det(B_ss)
        proba_cumsum[r] = cumsum
        write_subset_to_array(idx, s, r, num)
        s = next_combination(s, n)
        r += 1
    return proba_cumsum, idx


# Sampling procedure for `tau`-element volume sampling with respect to `B`
@jit('i4[:](f8[:], i4[:], i4)', nopython=True, cache=True)
def vs_sample(proba_cumsum, idx, tau):
    u = np.random.rand()
    r = np.searchsorted(proba_cumsum, u * proba_cumsum[-1])
    s = read_subset_from_array(idx, tau, r, proba_cumsum.size)
    return s


##########################################################################
########################### General sampling #############################
##########################################################################

# Preprocessing procedure for `tau`-element sampling (either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`) with respect to `B`
# Currently support only `tau <= 3`.
@jit('Tuple((f8[:], i4[:]))(f8[:,:], i4, i4)', nopython=True, cache=True)
def sampling_preprocess(B, tau, sampling):
    proba_cumsum = np.zeros(0)
    idx = np.zeros(0, dtype=np.int32)
    if sampling == SAMPLING_VOLUME:
        proba_cumsum, idx = vs_preprocess(B, tau)
    return proba_cumsum, idx


# Sampling procedure for `tau`-element sampling (either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`) with respect to `B`
@jit('i4[:](f8[:], i4[:], i4, i4, i4)', nopython=True, cache=True)
def sampling_sample(proba_cumsum, idx, n, tau, sampling):
    s = np.zeros(0, dtype=np.int32)
    if sampling == SAMPLING_UNIFORM:
        s = us_sample(n, tau)
    if sampling == SAMPLING_VOLUME:
        s = vs_sample(proba_cumsum, idx, tau)
    return s

##########################################################################
######################## Sparse volume sampling ##########################
##########################################################################

# Sparse preprocessing for 1-element volume sampling with respect to `B`
# `B` (of size `n x n`) is given by its CSR representation `(data, indices, indptr)`
# Assume that `indices` are sorted!
@jit('f8[:](i4, f8[:], i4[:], i4[:])', nopython=True, cache=True)
def sparse_1vs_preprocess(n, data, indices, indptr):
    diag_proba_cumsum = np.zeros(n)
    cumsum = 0
    for i in range(n):
        diag_i = csr_get_element(i, i, data, indices, indptr)
        cumsum += diag_i
        diag_proba_cumsum[i] = cumsum
    return diag_proba_cumsum

# Sparse sampling routine for 1-element volume sampling with respect to `B`
@jit('i4[:](f8[:])', nopython=True, cache=True)
def sparse_1vs_sample(diag_proba_cumsum):
    u = np.random.rand()
    i = np.searchsorted(diag_proba_cumsum, u * diag_proba_cumsum[-1])
    s = np.array([i], dtype=np.int32)
    return s


# Auxiliary function `P` that is used inside `sparse_2vs_sample`
@jit('f8(i4, i4, i4, f8[:], i4[:], i4[:], f8[:], i4[:], f8[:])', nopython=True, cache=True)
def _sparse_2vs_sample_P(i, j, k, data, indptr, diagptr, h_data, h_indptr, t):
    diag_i = data[diagptr[i]] if diagptr[i] < indptr[i + 1] else 0
    h_jk = h_data[h_indptr[i] + k] if k >= 0 and h_indptr[i] + k < h_indptr[i + 1] else 0
    return diag_i * (t[i] - t[j + 1]) - h_jk

# Sparse preprocessing for 2-element volume sampling with respect to `B`
# `B` (of size `n x n`) is given by its CSR representation `(data, indices, indptr)`
# Assume that `indices` are sorted!
@jit('Tuple((i4[:], f8[:], i4[:], f8[:], f8[:]))(i4, f8[:], i4[:], i4[:])', nopython=True, cache=True)
def sparse_2vs_preprocess(n, data, indices, indptr):
    # Compute the positions of diagonal elements
    diagptr = np.zeros(n, dtype=np.int32)
    for i in range(n):
        offset = np.searchsorted(indices[indptr[i]:indptr[i + 1]], i)
        diagptr[i] = indptr[i] + offset
    # Now everything to the right of the diagonal element is indexed by `diagptr[i]:indptr[i+1]`
    # Compute `h`
    h_size = 0
    for i in range(n):
        h_size += indptr[i + 1] - diagptr[i]
    h_data = np.zeros(h_size)
    h_indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        data_after_diag_i = data[diagptr[i]:indptr[i + 1]]
        h_indptr[i + 1] = h_indptr[i] + data_after_diag_i.size
        h_data[h_indptr[i]:h_indptr[i + 1]] = np.cumsum(data_after_diag_i**2)
    # Compute `t`
    t = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):  # from `n - 1` down to `0`
        diag_i = data[diagptr[i]] if diagptr[i] < indptr[i + 1] else 0
        t[i] = t[i + 1] + diag_i
    # Compute `q`
    q = np.zeros(n - 1)
    cumsum = 0
    for i in range(n - 1):
        cumsum += _sparse_2vs_sample_P(i, n - 1, h_indptr[i + 1] - h_indptr[i] - 1, data, indptr, diagptr, h_data, h_indptr, t)
        q[i] = cumsum
    return diagptr, h_data, h_indptr, t, q

# Binary search for finding `kl` inside `sparse_2vs_sample`
@jit('i4(f8, i4, i4, f8[:], i4[:], i4[:], i4[:], f8[:], i4[:], f8[:])', nopython=True, cache=True)
def _sparse_2vs_sample_kl(u2_tilde, i, n, data, indices, indptr, diagptr, h_data, h_indptr, t):
    left = 0
    right = indptr[i + 1] - diagptr[i]
    while left < right:
        k = (left + right) // 2
        ptr = diagptr[i] + k + 1
        jkp1 = indices[ptr] if ptr < indptr[i + 1] else n
        P = _sparse_2vs_sample_P(i, jkp1 - 1, k, data, indptr, diagptr, h_data, h_indptr, t)
        if u2_tilde > P:
            left = k + 1
        else:
            right = k
    return left

# Binary search for finding `j` inside `sparse_2vs_sample`
@jit('i4(f8, i4, i4, i4, f8[:], i4[:], i4[:], i4[:], f8[:], i4[:], f8[:])', nopython=True, cache=True)
def _sparse_2vs_sample_j(u2_tilde, i, kl, n, data, indices, indptr, diagptr, h_data, h_indptr, t):
    ptr = diagptr[i] + kl
    assert(ptr < indptr[i + 1])
    left = indices[ptr]
    right = indices[ptr + 1] if ptr + 1 < indptr[i + 1] else n
    while left < right:
        j = (left + right) // 2
        P = _sparse_2vs_sample_P(i, j, kl, data, indptr, diagptr, h_data, h_indptr, t)
        if u2_tilde > P:
            left = j + 1
        else:
            right = j
    return left

# Sparse sampling routine for 2-element volume sampling with respect to `B`
# `B` (of size `n x n`) is given by its CSR representation `(data, indices, indptr)`
# Assume that `indices` are sorted!
@jit('i4[:](i4, f8[:], i4[:], i4[:], i4[:], f8[:], i4[:], f8[:], f8[:])', nopython=True, cache=True)
def sparse_2vs_sample(n, data, indices, indptr, diagptr, h_data, h_indptr, t, q):
    u1, u2 = np.random.rand(2)
    i = np.searchsorted(q, u1 * q[-1])
    u2_tilde = u2 * _sparse_2vs_sample_P(i, n - 1, h_indptr[i + 1] - h_indptr[i] - 1, data, indptr, diagptr, h_data, h_indptr, t)
    kl = _sparse_2vs_sample_kl(u2_tilde, i, n, data, indices, indptr, diagptr, h_data, h_indptr, t)
    j = _sparse_2vs_sample_j(u2_tilde, i, kl, n, data, indices, indptr, diagptr, h_data, h_indptr, t)
    s = np.array([i, j], dtype=np.int32)
    return s


##########################################################################
######################## Sparse general sampling #########################
##########################################################################

# Preprocessing procedure for sparse `tau`-element sampling (either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`) with respect to `B`
# `B` (of size `n x n`) is given by its CSR representation `(data, indices, indptr)`
# For volume sampling currently support only `tau <= 2`.
@jit('Tuple((f8[:], i4[:], f8[:], i4[:], f8[:], f8[:]))(i4, f8[:], i4[:], i4[:], i4, i4)', nopython=True, cache=True)
def sparse_sampling_preprocess(n, data, indices, indptr, tau, sampling):
    diag_proba_cumsum = np.zeros(0)
    diagptr = np.zeros(0, dtype=np.int32)
    h_data = np.zeros(0)
    h_indptr = np.zeros(0, dtype=np.int32)
    t = np.zeros(0)
    q = np.zeros(0)
    if sampling == SAMPLING_VOLUME:
        assert(tau <= 2)
        if tau == 1:
            diag_proba_cumsum = sparse_1vs_preprocess(n, data, indices, indptr)
        if tau == 2:
            diagptr, h_data, h_indptr, t, q = sparse_2vs_preprocess(n, data, indices, indptr)
    return diag_proba_cumsum, diagptr, h_data, h_indptr, t, q


# Sampling procedure for sparse `tau`-element sampling (either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`) with respect to `B`
# `B` (of size `n x n`) is given by its CSR representation `(data, indices, indptr)`
# For volume sampling currently support only `tau <= 2`.
@jit('i4[:](i4, f8[:], i4[:], i4[:], f8[:], i4[:], f8[:], i4[:], f8[:], f8[:], i4, i4)', nopython=True, cache=True)
def sparse_sampling_sample(n, data, indices, indptr, diag_proba_cumsum, diagptr, h_data, h_indptr, t, q, tau, sampling):
    s = np.zeros(0, dtype=np.int32)
    if sampling == SAMPLING_UNIFORM:
        s = us_sample(n, tau)
    if sampling == SAMPLING_VOLUME:
        assert(tau <= 2)
        if tau == 1:
            s = sparse_1vs_sample(diag_proba_cumsum)
        if tau == 2:
            s = sparse_2vs_sample(n, data, indices, indptr, diagptr, h_data, h_indptr, t, q)
    return s
