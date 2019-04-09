#!/usr/bin/env python3

import numpy as np
from time import time
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
import argparse
from methods import *
from sampling import *
from functions import *
from generation import *

##########################################################################
# Define global constants
EXPERIMENT_QUADRATIC = 0
EXPERIMENT_HUBER = 1
EXPERIMENT_HUBER_SPARSE = 2
EXPERIMENT_LOGISTIC = 3
##########################################################################

##########################################################################
########################## Auxiliary functions ###########################
##########################################################################

# Compute explicitly eigenvalues (in decreasing order) of matrix `B = L A^T A + gamma I`
def get_eigvals_B(A, gamma, mu, function):
    L = get_smoothness_constant(function, mu)
    n = A.shape[1]
    B = L * A.T.dot(A) + gamma * np.eye(n)
    eigvals = np.sort(np.linalg.eigvalsh(B))[::-1]
    return eigvals

# Compute theoretical acceleration rate given eigenvalues (in decreasing order) of `B`
def get_theory_acc_rate(eigvals, tau):
    return np.sum(eigvals) / np.sum(eigvals[tau-1:])

##########################################################################
####################### General scheme of experiment #####################
##########################################################################

# General scheme of experiment: one particular run
def run1(experiment, params, seed=31415):
    methods = [
        {'tau': 1, 'sampling': SAMPLING_VOLUME},
        {'tau': params['tau'], 'sampling': SAMPLING_UNIFORM},
        {'tau': params['tau'], 'sampling': SAMPLING_VOLUME},
    ]

    np.random.seed(seed)

    if experiment == EXPERIMENT_LOGISTIC:
        A, b = load_svmlight_file('datasets/%s' % params['dataset'])
        A = A.toarray()
        gamma = 1
        mu = 0  # dummy variable
        f_opt = compute_sepfunc_f_opt(A, b, gamma, mu, FUNCTION_LOGISTIC)

    results = [None for method in methods]
    for method_idx, method in enumerate(methods):
        results[method_idx] = defaultdict(list)
    for i_run in range(params['n_runs']):
        # Generate a problem instance
        if experiment == EXPERIMENT_QUADRATIC:
            A, b, f_opt = generate_random_quadratic_instance(params['n'], params['eigval1'], params['eigval2'])
        if experiment == EXPERIMENT_HUBER:
            A, b, gamma, f_opt = generate_random_separable_instance(params['m'], params['n'], params['eigval1'], params['eigval2'], params['mu'], FUNCTION_HUBER)
        if experiment == EXPERIMENT_HUBER_SPARSE:
            A, b, gamma, f_opt = generate_random_sparse_separable_instance(params['m'], params['n'], params['p'], params['eigval1'], params['eigval2'], params['mu'], FUNCTION_HUBER)

        ######
        # Run each method
        for method_idx, method in enumerate(methods):
            t_start = time()

            if experiment == EXPERIMENT_QUADRATIC:
                x_final, success, n_iters = cd_quadratic(
                    A, b, f_opt, params['eps'], params['max_iter'], method['tau'], method['sampling']
                )
            if experiment == EXPERIMENT_HUBER:
                x_final, success, n_iters = cd_separable(
                    A, b, gamma, params['mu'], f_opt, params['eps'], params['max_iter'], method['tau'], FUNCTION_HUBER, method['sampling']
                )
            if experiment == EXPERIMENT_HUBER_SPARSE:
                x_final, success, n_iters = cd_sparse_separable(
                    A, b, gamma, params['mu'], f_opt, params['eps'], params['max_iter'],
                    method['tau'], FUNCTION_HUBER, method['sampling']
                )
            if experiment == EXPERIMENT_LOGISTIC:
                x_final, success, n_iters = cd_separable(
                    A, b, gamma, mu, f_opt, params['eps'], params['max_iter'],
                    method['tau'], FUNCTION_LOGISTIC, method['sampling'])

            total_time = time() - t_start

            results[method_idx]['success'].append(success)
            results[method_idx]['total_time'].append(total_time)
            results[method_idx]['n_iters'].append(n_iters)

    ######
    iter_format = '%-{w}d'  # default iter format (integer)
    time_decimals = 1  # default number of decimals in time column
    if experiment == EXPERIMENT_QUADRATIC:
        param_format = '%-{w}d %-{w}d '
        param_args = (params['n'], params['eigval1'] / params['eigval2'])
    if experiment in [EXPERIMENT_HUBER, EXPERIMENT_HUBER_SPARSE]:
        param_format = '%-{w}d %-{w}d %-{w}d '
        param_args = (params['m'], params['n'], params['eigval1'] / params['eigval2'])
    if experiment == EXPERIMENT_LOGISTIC:
        param_format = '%-{w1}s %-{w}d '
        param_args = (params['dataset'], params['tau'])
        iter_format = '%-{w}.1f'
        time_decimals = 2

    row_format = (
        param_format +
        iter_format + ' %-{w}.{td}f ' +
        iter_format + ' %-{w}.1f %-{w}.{td}f ' +
        iter_format + ' %-{w}d %-{w}d %-{w}.2f'
    ).format(w=params['col_width'], w1=params['col_width1'], td=time_decimals)

    # Compute theoretical acceleration rate
    if experiment == EXPERIMENT_QUADRATIC:
        eigvals = np.concatenate(([params['eigval1'], params['eigval2']], np.ones(params['n'] - 2)))
    if experiment in [EXPERIMENT_HUBER, EXPERIMENT_HUBER_SPARSE]:
        q = min(params['m'], params['n'])
        eigvals = np.concatenate(([params['eigval1'], params['eigval2']], np.ones(q - 2), np.zeros(params['n'] - q))) + gamma
    if experiment == EXPERIMENT_LOGISTIC:
        eigvals = get_eigvals_B(A, gamma, mu, FUNCTION_LOGISTIC)
    theory_acc = get_theory_acc_rate(eigvals, params['tau'])

    # Compute values of columns
    rcd_it = np.median(results[0]['n_iters']) / 1000
    rcd_t = np.median(results[0]['total_time'])

    sdna_it = np.median(results[1]['n_iters']) / 1000
    sdna_acc = np.median(np.asarray(results[0]['n_iters']) / results[1]['n_iters'])
    sdna_t = np.median(results[1]['total_time'])

    rcdvs_it = np.median(results[2]['n_iters']) / 1000
    rcdvs_acc = np.median(np.asarray(results[0]['n_iters']) / results[2]['n_iters'])
    rcdvs_per = rcdvs_acc / theory_acc * 100
    rcdvs_t = np.median(results[2]['total_time'])

    # Print row
    row_output = row_format % (
        *param_args,
        rcd_it, rcd_t,
        sdna_it, sdna_acc, sdna_t,
        rcdvs_it, rcdvs_acc, rcdvs_per, rcdvs_t
    )
    print(row_output)

    return results

# General scheme of experiment
def run_general(experiment, params):
    #####
    # Print table header
    print('eps=%f, max_iter=%d, n_runs=%d' % (
        params['eps'], params['max_iter'], params['n_runs']
    ))

    if experiment == EXPERIMENT_QUADRATIC:
        params_format = '%-{w}s %-{w}s '
        params_args = ('n', 'l1/l2')
    if experiment in [EXPERIMENT_HUBER, EXPERIMENT_HUBER_SPARSE]:
        params_format = '%-{w}s %-{w}s %-{w}s '
        params_args = ('m', 'n', 'l1/l2')
    if experiment == EXPERIMENT_LOGISTIC:
        params_format = '%-{w1}s %-{w}s '
        params_args = ('Data', 'tau')

    header_format = (
        params_format +
        '%-{w}s %-{w}s ' +
        '%-{w}s %-{w}s %-{w}s ' +
        '%-{w}s %-{w}s %-{w}s %-{w}s'
    ).format(w=params['col_width'], w1=params['col_width1'])
    header_output = header_format % (
        *params_args,
        'It', 'T',
        'It', 'Acc', 'T',
        'It', 'Acc', '%', 'T'
    )
    print(header_output)
    #####

    if experiment == EXPERIMENT_QUADRATIC:
        for n in params['n_values']:
            for eig1deig2 in params['eig1deig2_values']:
                params['n'] = n
                params['eigval1'] = params['eigval2'] * eig1deig2
                run1(experiment, params)
            print()
    if experiment == EXPERIMENT_HUBER:
        for (m, n) in params['mn_values']:
            for eig1deig2 in params['eig1deig2_values']:
                params['m'] = m
                params['n'] = n
                params['eigval1'] = params['eigval2'] * eig1deig2
                run1(experiment, params)
            print()
    if experiment == EXPERIMENT_HUBER_SPARSE:
        for (m, n, p) in params['mnp_values']:
            for eig1deig2 in params['eig1deig2_values']:
                params['m'] = m
                params['n'] = n
                params['p'] = p
                params['eigval1'] = params['eigval2'] * eig1deig2
                run1(experiment, params)
            print()
    if experiment == EXPERIMENT_LOGISTIC:
        for dataset in params['datasets']:
            for tau in params['taus']:
                params['dataset'] = dataset
                params['tau'] = tau
                run1(experiment, params)
            print()

##########################################################################
######################## Particular experiments ##########################
##########################################################################

# Experiment: Quadratic function
def run_quadratic():
    params = {
        'n_values': [400, 800, 1600, 3200],
        'eig1deig2_values': [4, 16, 64, 256, 1024],
        'eigval2': 100,
        'tau': 2,
        'eps': 0.01,
        'max_iter': 100000000,
        'n_runs': 10,
        'col_width': 6,  # width of column (only for printing purposes)
        'col_width1': None  # dummy
    }
    run_general(EXPERIMENT_QUADRATIC, params)

# Experiment: Huber function
def run_huber():
    params = {
        'mn_values': [(400, 800), (800, 400), (800, 1600), (1600, 800)],
        'eig1deig2_values': [4, 16, 64, 256, 1024],
        'eigval2': 100,
        'tau': 2,
        'eps': 0.01,
        'mu': 0.01,
        'max_iter': 100000000,
        'n_runs': 10,
        'col_width': 6,  # width of column (only for printing purposes)
        'col_width1': None  # dummy
    }
    run_general(EXPERIMENT_HUBER, params)


# Experiment: Huber function on sparse data
def run_huber_sparse():
    params = {
        'mnp_values': [(8000, 16000, 50), (16000, 8000, 50), (16000, 32000, 70), (32000, 16000, 70)],
        'eig1deig2_values': [64, 256, 1024, 4096, 16384],
        'eigval2': 100,
        'tau': 2,
        'eps': 0.01,
        'mu': 0.01,
        'max_iter': 1000000000,
        'n_runs': 10,
        'col_width': 8,  # width of column (only for printing purposes)
        'col_width1': None  # dummy
    }
    run_general(EXPERIMENT_HUBER_SPARSE, params)


# Experiment: Logistic regression
def run_logistic():
    params = {
        'datasets': ['breast-cancer_scale', 'phishing', 'a9a'],
        'taus': [2, 3, 4],
        'eps': 0.01,
        'max_iter': 10000000,
        'n_runs': 1,
        'col_width': 8,  # width of column (only for printing purposes)
        'col_width1': 25  # width of a big column (only for printing purposes)
    }
    run_general(EXPERIMENT_LOGISTIC, params)

##########################################################################
########################## Info about datasets ###########################
##########################################################################

# Print info about datasets
def print_data_info(function=FUNCTION_LOGISTIC, mu=0, col_width=8, col_width1=25):
    datasets = ['breast-cancer_scale', 'phishing', 'a9a']

    header_format = (
        '%-{w1}s %-{w}s %-{w}s' +
        '%-{w}s %-{w}s %-{w}s %-{w}s' +
        '%-{w}s %-{w}s %-{w}s'
    ).format(w=col_width, w1=col_width1)
    header_output = header_format % (
        'Data', 'm', 'n',
        'eig1', 'eig2', 'eig3', 'eig4',
        'rate2', 'rate3', 'rate4'
    )
    print(header_output)

    for dataset in datasets:
        # Load data
        A, b = load_svmlight_file('datasets/%s' % dataset)
        A = A.toarray()
        gamma = 1 if function == FUNCTION_LOGISTIC else 0

        # Estimate eigenvalues and acceleration rates
        eigvals = get_eigvals_B(A, gamma, mu, function)
        rates = []
        for tau in [2, 3, 4]:
            rate = get_theory_acc_rate(eigvals, tau)
            rates.append(rate)
        m, n = A.shape

        # Print
        row_format = (
            '%-{w1}s %-{w}d %-{w}d' +
            '%-{w}d %-{w}d %-{w}d %-{w}d' +
            '%-{w}.1f %-{w}.1f %-{w}.1f'
        ).format(w=col_width, w1=col_width1)
        row_output = row_format % (
            dataset, m, n,
            eigvals[0], eigvals[1], eigvals[2], eigvals[3],
            rates[0], rates[1], rates[2]
        )
        print(row_output)


##########################################################################
################################ Startup #################################
##########################################################################

# Main function that is run on startup
def startup():
    # Prepare
    actions_ref = {
        'run_quadratic': run_quadratic,
        'run_huber': run_huber,
        'run_huber_sparse': run_huber_sparse,
        'run_logistic': run_logistic,
        'print_data_info': print_data_info
    }
    actions = list(actions_ref.keys())
    action_help = 'Action (%s)' % ', '.join(actions)

    # Run argument parser
    parser = argparse.ArgumentParser(description='Experiments for the RCDVS paper (A. Rodomanov, D. Kropotov)')
    parser.add_argument('-a','--action', help=action_help, required=True)
    args = vars(parser.parse_args())

    # Check correctness
    if not args['action'] in actions:
        raise "Wrong `action` specified"

    # Run corresponding action
    actions_ref[args['action']]()

###################################################
if __name__ == '__main__':
    startup()
