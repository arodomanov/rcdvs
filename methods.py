import numpy as np
from numba import jit
from sampling import *
from auxiliary import *
from functions import *


####################################################################################
# Coordinate descent for the quadratic function `f(x) = (1/2) <Ax, x> - <b, x>`
# f_opt = optimal function value for termination criterion
# x0 = 0
# Sampling is either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`
@jit('Tuple((f8[:], b1, i4))(f8[:,:], f8[:], f8, f8, i4, i4, i4)', nopython=True, cache=True)
def cd_quadratic(A, b, f_opt, eps, max_iter, tau, sampling):
    np.random.seed(31415)  # For consistency between different runs
    n = A.shape[0]
    x = np.zeros(n)
    y = -b  # Define `y := A x - b`
    func_val = 0
    success = False
    proba_cumsum, idx = sampling_preprocess(A, tau, sampling)
    for cur_iter in range(max_iter):
        if func_val - f_opt < eps:
            success = True
            break
        s = sampling_sample(proba_cumsum, idx, n, tau, sampling)
        g_s = y[s]
        A_ss = get_principal_submatrix(A, s)
        delta_x_s = -np.linalg.solve(A_ss, g_s)
        delta_y = matvec_ind(A, delta_x_s, s)
        delta_y_s = delta_y[s]
        delta_f = np.dot(delta_x_s, g_s + 0.5 * delta_y_s)
        func_val += delta_f
        x[s] += delta_x_s
        y += delta_y
    return x, success, cur_iter


# Coordinate descent for the separable function `f(x) = sum_i phi_i((A x)_i) + (gamma/2) ||x||^2`
# f_opt = optimal function value for termination criterion
# `b` is a vector used inside `phi`
# `x0 = 0`
# `phi` is specified by `function`
# Sampling is either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`
@jit('Tuple((f8[:], b1, i4))(f8[:,:], f8[:], f8, f8, f8, f8, i4, i4, i4, i4)', nopython=True, cache=True)
def cd_separable(A, b, gamma, mu, f_opt, eps, max_iter, tau, function, sampling):
    np.random.seed(31415)  # For consistency between different runs
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(m)  # Define `y := A x`
    success = False
    L = get_smoothness_constant(function, mu)
    if tau == 1:
        B = np.diag(L * np.sum(A**2, axis=0) + gamma)
    else:
        B = L * A.T.dot(A) + gamma * np.eye(n)
    proba_cumsum, idx = sampling_preprocess(B, tau, sampling)
    for cur_iter in range(max_iter):
        func_val = np.sum(sepfunc_values(y, b, mu, function)) + (gamma / 2) * np.dot(x, x)
        if func_val - f_opt < eps:
            success = True
            break
        s = sampling_sample(proba_cumsum, idx, n, tau, sampling)
        grad_phi = sepfunc_derivatives(y, b, mu, function)
        g_s = matvec_T_ind(A, grad_phi, s) + gamma * x[s]
        B_ss = get_principal_submatrix(B, s)
        delta_x_s = -np.linalg.pinv(B_ss).dot(g_s)
        delta_y = matvec_ind(A, delta_x_s, s)
        x[s] += delta_x_s
        y += delta_y
    return x, success, cur_iter


# Coordinate descent for the separable function `f(x) = sum_i phi_i((A x)_i) + (gamma/2) ||x||^2`
# `A`: sparse CSC matrix
# `b` is a dense vector used inside `phi`
# f_opt = optimal function value for termination criterion
# `x0 = 0`
# `phi` is specified by `function`
# Sampling is either `SAMPLING_UNIFORM` or `SAMPLING_VOLUME`
# Currently support only `tau <= 2` for volume sampling
def cd_sparse_separable(A, b, gamma, mu, f_opt, eps, max_iter, tau, function, sampling):
    m, n = A.shape
    L = get_smoothness_constant(function, mu)
    if tau == 1:
        diag_B = L * np.asarray((A.multiply(A)).sum(axis=0)).reshape(-1) + gamma
        B = ssp.diags(diag_B, format='csr')
    else:
        B = L * (A.T * A) + gamma * ssp.eye(n)
        B.sort_indices()
    assert(ssp.isspmatrix_csr(B))
    return cd_sparse_separable_core(m, n, A.data, A.indices, A.indptr, B.data, B.indices, B.indptr,
        b, gamma, mu, f_opt, eps, max_iter, tau, function, sampling)


@jit('Tuple((f8[:], b1, i4))(i4, i4, f8[:], i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], f8, f8, f8, f8, i4, i4, i4, i4)', nopython=True, cache=True)
def cd_sparse_separable_core(m, n, A_data, A_indices, A_indptr, B_data, B_indices, B_indptr,
        b, gamma, mu, f_opt, eps, max_iter, tau, function, sampling):
    np.random.seed(31415)  # For consistency between different runs
    assert(tau <= 2)
    x = np.zeros(n)
    y = np.zeros(m)  # Define `y := A x`
    success = False
    func_val = np.sum(sepfunc_values(y, b, mu, function)) + (gamma / 2) * np.dot(x, x)
    grad_phi = sepfunc_derivatives(y, b, mu, function)
    diag_proba_cumsum, diagptr, h_data, h_indptr, t, q = sparse_sampling_preprocess(n, B_data, B_indices, B_indptr, tau, sampling)
    for cur_iter in range(max_iter):
        if func_val - f_opt < eps:
            success = True
            break
        s = sparse_sampling_sample(n, B_data, B_indices, B_indptr, diag_proba_cumsum, diagptr, h_data, h_indptr, t, q, tau, sampling)
        g_s = csc_matvec_T_ind(A_data, A_indices, A_indptr, grad_phi, s) + gamma * x[s]
        B_ss = csr_get_principal_submatrix(B_data, B_indices, B_indptr, s)
        delta_x_s = -np.linalg.pinv(B_ss).dot(g_s)
        delta_y_sy, sy = csc_matvec_ind(m, A_data, A_indices, A_indptr, delta_x_s, s)
        y_sy = y[sy]
        y_new_sy = y_sy + delta_y_sy
        delta_func_val = np.sum(sepfunc_values(y_new_sy, b[sy], mu, function) - sepfunc_values(y_sy, b[sy], mu, function))
        delta_func_val += (gamma / 2) * (2 * np.dot(x[s], delta_x_s) + np.dot(delta_x_s, delta_x_s))
        delta_grad_phi_sy = sepfunc_derivatives(y_new_sy, b[sy], mu, function) - sepfunc_derivatives(y_sy, b[sy], mu, function)
        func_val += delta_func_val
        grad_phi[sy] += delta_grad_phi_sy
        x[s] += delta_x_s
        y[sy] += delta_y_sy
    return x, success, cur_iter
