# Description

The Python 3 code that reproduces the experimental results from the paper
```
A Randomized Coordinate Descent Method with Volume Sampling (A. Rodomanov, D. Kropotov)
```

# Installation

To be able to execute the code, make sure you have the following Python packages installed:
1. `numpy` and `scipy` (for basic linear algebra operations)
2. `numba` (for fast execution of the code)
3. `sklearn` (for loading the data for the logistic regression experiment)

In addition to that, for the logistic regression experiment, you need to download the following datasets from the [LIBSVM website](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) and put them inside the `datasets` folder:
1. `breast-cancer_scale` ([download](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale))
2. `phishing` ([download](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing))
3. `a9a` ([download](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a))

# Usage

**Note**: The code has been tested on Ubuntu Linux but it should work on any other operating system that has Python 3 installed.

The main script is `experiments.py`. You can run all the experiments from the terminal using the following commands:
1. Quadratic function: `./experiments.py -a run_quadratic`.
2. Huber function: `./experiments.py -a run_huber`.
3. Huber function on sparse data: `./experiments.py -a run_huber_sparse`.
4. Logistic regression: `./experiments.py -a run_logistic`.

You can also print the information about the real data for the logistic regression experiment using `./experiments.py -a print_data_info`.
