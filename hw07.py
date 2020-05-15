# 1. Try QP method to solve lasso problem, write your code to conduct simulation, use cvxpy or cvxopt package.
# Computing trade-off curves is a common use of parameters. The example below computes a trade-off curve for a LASSO problem.

import cvxpy as cvx
import numpy
import matplotlib.pyplot as plt
import cvxpy as cp
def generate_data(n=100, p=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_true = np.random.randn(p)
    idxs = np.random.choice(range(p), int((1-density)*p), replace=False)
    for idx in idxs:
        beta_true[idx] = 0
    X = np.random.randn(n,p)
    Y = X.dot(beta_true) + np.random.normal(0, sigma, size=n)
    return X, Y, beta_true


# 2. Try LQA method to solve lasso problem, write your code to conduct simulation. 

# Analyze the stock return data, test the result of different variable selection methods for index tracking.