import numpy as np
from numba import jit
import scipy.sparse as ssp

# Compute `A_s v` where `s` is the index set specifying columns of `A`, `v` is a vector
# This is faster than direct numpy code `A[:,s].dot(v)`
@jit('f8[:](f8[:,:], f8[:], i4[:])', nopython=True)
def matvec_ind(A, v, s):
    m = A.shape[0]
    res = np.zeros(m)
    for i in range(m):
        for k in range(s.size):
            res[i] += A[i, s[k]] * v[k]
    return res

# Compute `A_s^T w` where `s` is the index set specifying columns of `A`, `w` is a vector
# This is faster than direct numpy code `A[:, s].T.dot(w)`
@jit('f8[:](f8[:,:], f8[:], i4[:])', nopython=True)
def matvec_T_ind(A, w, s):
    m = A.shape[0]
    res = np.zeros(s.size)
    for k in range(s.size):
        for i in range(m):
            res[k] += A[i, s[k]] * w[i]
    return res


# Return the principal submatrix `B_{S \times S}` where the row/column indices are specified by `s`
@jit('f8[:,:](f8[:,:], i4[:])', nopython=True)
def get_principal_submatrix(B, s):
    tau = s.size
    res = np.zeros((tau, tau))
    for idx1 in range(tau):
        for idx2 in range(tau):
            res[idx1, idx2] = B[s[idx1], s[idx2]]
    return res


# Construct a sparse vector `v` (`n-by-1` CSC matrix) which has the non-zero part `v_s` at the indices from `s`
def sparse_vector(v_s, s, n):
    indices = s
    data = v_s
    indptr = np.array([0, s.size], dtype=np.int32)
    v = ssp.csc_matrix((data, indices, indptr), shape=(n, 1))
    return v

# Compute `z := A_s v` where `s` is the index set specifying columns of a *sparse* matrix `A` with `m` rows, `v` is a *dense* vector
# `A` is given by its CSC format `(data, indices, indptr)`
# Return `z` as a sparse vector represented by `z_data` (non-zeros) and `z_indices` (positions of non-zeros)
# Some of the elements of `z` might be zero after this routine. Perhaps, some additional zero-removal routine should be used.
# Complexity: O(size(s) * nnz(A_s)) (perhaps might be improved, but we only need it for small `s`, so we leave it as it is)
# Assume `indices` are sorted!
@jit('Tuple((f8[:], i4[:]))(i4, f8[:], i4[:], i4[:], f8[:], i4[:])', nopython=True)
def csc_matvec_ind(m, data, indices, indptr, v, s):
    tau = s.size
    ptr = np.zeros(tau, dtype=np.int32)
    z_size = 0
    for t in range(tau):
        ptr[t] = indptr[s[t]]
        z_size += indptr[s[t] + 1] - indptr[s[t]]
    # Now `z_size = nnz(A_s)`
    z_data = np.zeros(z_size)
    z_indices = np.zeros(z_size, dtype=np.int32)
    z_ptr = 0
    while True:
        i = m
        for t in range(tau):
            if ptr[t] < indptr[s[t] + 1] and indices[ptr[t]] < i:
                i = indices[ptr[t]]
        if i == m:  # not found
            break
        z_indices[z_ptr] = i
        for t in range(tau):
            if ptr[t] < indptr[s[t] + 1] and indices[ptr[t]] == i:
                z_data[z_ptr] += data[ptr[t]] * v[t]
                ptr[t] += 1
        z_ptr += 1
    return z_data[:z_ptr], z_indices[:z_ptr]


# Compute `z := (A_s)^T w` where `s` is the index set specifying columns of a *sparse* matrix `A`, `w` is a *dense* vector
# `A` is given by its CSC format `(data, indices, indptr)`
# Return `z` as a *dense* vector
# Some of the elements of `z` might be zero after this routine. Perhaps, some additional zero-removal routine should be used.
# Complexity: O(nnz(A_s))
@jit('f8[:](f8[:], i4[:], i4[:], f8[:], i4[:])', nopython=True)
def csc_matvec_T_ind(data, indices, indptr, w, s):
    tau = s.size
    z = np.zeros(tau)
    for t in range(tau):
        for ptr in range(indptr[s[t]], indptr[s[t] + 1]):
            z[t] += data[ptr] * w[indices[ptr]]
    return z


# Return `B[i, j]` where `data, indices, indptr` is a CSR representation of `B` (of size `n x n`)
# Assume `indices` are sorted!
# Complexity: O(log n)
@jit('f8(i4, i4, f8[:], i4[:], i4[:])', nopython=True)
def csr_get_element(i, j, data, indices, indptr):
    # Use binary search to find `j` inside `indices[indptr[i]:indptr[i+1]]`
    leftptr = indptr[i]
    rightptr = indptr[i + 1]
    while leftptr < rightptr:
        ptr = (leftptr + rightptr) // 2
        if indices[ptr] < j:
            leftptr = ptr + 1
        else:
            rightptr = ptr
    if leftptr < indptr[i + 1] and indices[leftptr] == j:
        return data[leftptr]
    return 0  # If not found


# Return the principal submatrix `B_{s x s}` of the symmetric matrix `B` (of size `n x n`)
# `B` is given by its CSR format `(data, indices, indptr)`
# Return `B_{s x s}` as a dense submatrix (assume `s` is small, so sparsity does not matter)
# assume `indices` are sorted!
# Complexity: O(size(s)^2 log n)
@jit('f8[:,:](f8[:], i4[:], i4[:], i4[:])', nopython=True)
def csr_get_principal_submatrix(data, indices, indptr, s):
    tau = s.size
    B_ss = np.zeros((tau, tau))
    for t1 in range(tau):
        for t2 in range(t1, tau):
            elem = csr_get_element(s[t1], s[t2], data, indices, indptr)
            B_ss[t1, t2] = elem
            B_ss[t2, t1] = elem
    return B_ss


# Given a subset `s` of `[n] \choose \tau`, return the next subset in the lexicographic order
# `s` is given as array of size `\tau` with `s[0] < ... < s[tau - 1]`
# If `s` is already the last, return the empty array.
@jit('i4[:](i4[:], i4)', nopython=True)
def next_combination(s, n):
    tau = s.size
    idx = tau - 1
    while idx >= 0 and s[idx] == n - tau + idx:
        idx -= 1
    if idx == -1:
        return np.zeros(0, dtype=np.int32)
    val = s[idx]
    for i in range(idx, tau):
        s[i] = val + 1
        val += 1
    return s

# Compute the binomial coefficient `n \choose \tau`
# Assume `n` and `tau` are moderate, so we do not care about any possible overflow
@jit('i4(i4, i4)', nopython=True)
def binom(n, tau):
    numer = 1
    denom = 1
    for i in range(tau):
        numer *= n - tau + i + 1
        denom *= i + 1
    return numer / denom

# Store subset `s` of indices inside `idx` at position `r`
# `idx` is assumed to have size `tau x base_size` and should be thought of as a matrix; `s` is then stored in the `i`-th column of `idx`
@jit('void(i4[:], i4[:], i4, i4)', nopython=True)
def write_subset_to_array(idx, s, r, base_size):
    for i in range(s.size):
        idx[base_size * i + r] = s[i]

# Read the `tau`-element subset `s` from the `r`-th position of `idx`
# This is the inverse routine to `write_subset_to_array`
@jit('i4[:](i4[:], i4, i4, i4)', nopython=True)
def read_subset_from_array(idx, tau, r, base_size):
    s = np.zeros(tau, dtype=np.int32)
    for i in range(tau):
        s[i] = idx[base_size * i + r]
    return s
