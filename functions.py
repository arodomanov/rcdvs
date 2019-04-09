import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from numba import jit

##########################################################################
# Define global constants
FUNCTION_LEAST_SQUARES = 1
FUNCTION_LOGISTIC = 2
FUNCTION_HUBER = 3
##########################################################################

@jit('f8[:](f8[:], f8[:])', nopython=True, cache=True)
def least_squares_values(y, b):
    return 0.5 * (y - b)**2

@jit('f8[:](f8[:], f8[:])', nopython=True, cache=True)
def least_squares_derivatives(y, b):
    return y - b

# Implemement sum_i ln(1 + exp(-c_i * x_i))
@jit('f8[:](f8[:], f8[:])', nopython=True, cache=True)
def logistic_values(y, b):
    return np.logaddexp(np.zeros_like(y), -b * y)

@jit('f8[:](f8[:], f8[:])', nopython=True, cache=True)
def logistic_derivatives(y, b):
    return -b / (1 + np.exp(b * y))

@jit('f8[:](f8[:], f8)', nopython=True, cache=True)
def standard_huber_values(z, mu):
    res = np.zeros_like(z)
    mask = np.abs(z) <= mu
    res[mask] = z[mask]**2 / (2 * mu)
    res[~mask] = np.abs(z[~mask]) - mu/2
    return res

@jit('f8[:](f8[:], f8)', nopython=True, cache=True)
def standard_huber_derivatives(z, mu):
    res = np.zeros_like(z)
    mask = np.abs(z) <= mu
    res[mask] = z[mask] / mu
    res[~mask] = np.sign(z[~mask])
    return res

@jit('f8[:](f8[:], f8[:], f8)', nopython=True, cache=True)
def huber_values(y, b, mu):
    return standard_huber_values(y - b, mu)

@jit('f8[:](f8[:], f8[:], f8)', nopython=True, cache=True)
def huber_derivatives(y, b, mu):
    return standard_huber_derivatives(y - b, mu)

@jit('f8(i4, f8)', nopython=True, cache=True)
def get_smoothness_constant(function, mu):
    L = 0
    if function == FUNCTION_LEAST_SQUARES:
        L = 1
    if function == FUNCTION_LOGISTIC:
        L = 0.25
    if function == FUNCTION_HUBER:
        L = 1 / mu
    return L

@jit('f8[:](f8[:], f8[:], f8, i4)', nopython=True, cache=True)
def sepfunc_values(y, b, mu, function):
    values = np.zeros(0)
    if function == FUNCTION_LEAST_SQUARES:
        values = least_squares_values(y, b)
    if function == FUNCTION_LOGISTIC:
        values = logistic_values(y, b)
    if function == FUNCTION_HUBER:
        values = huber_values(y, b, mu)
    return values

@jit('f8[:](f8[:], f8[:], f8, i4)', nopython=True, cache=True)
def sepfunc_derivatives(y, b, mu, function):
    derivatives = np.zeros(0)
    if function == FUNCTION_LEAST_SQUARES:
        derivatives = least_squares_derivatives(y, b)
    if function == FUNCTION_LOGISTIC:
        derivatives = logistic_derivatives(y, b)
    if function == FUNCTION_HUBER:
        derivatives = huber_derivatives(y, b, mu)
    return derivatives

# Compute the optimal value of a given separable function using some numerical method
def compute_sepfunc_f_opt(A, b, gamma, mu, function):
    n = A.shape[1]
    func = lambda x : np.sum(sepfunc_values(A.dot(x), b, mu, function)) + (gamma / 2) * np.dot(x, x)
    grad = lambda x : A.T.dot(sepfunc_derivatives(A.dot(x), b, mu, function)) + gamma * x
    res = minimize(func, x0=np.zeros(n), jac=grad, method='L-BFGS-B', tol=1e-16)
    f_opt = func(res.x)
    return f_opt
