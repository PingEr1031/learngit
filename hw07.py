# 1. Try QP method to solve lasso problem, write your code to conduct simulation, use cvxpy or cvxopt package.
# Computing trade-off curves is a common use of parameters. The example below computes a trade-off curve for a LASSO problem.

import cvxpy as cvx
import numpy as np
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

pip install cvxopt
from cvxopt  import solvers, matrix
def QP(X, Y, t):
    n, p = X.shape

    X_zero = np.zeros_like(X)
    X_c = np.hstack((X, X_zero))
    
    '''
    min 1/2 x^T P x + q^T x
    s.t. Gx <= h
         Ax = b
    '''
    
    # 构造P
    P = X_c.T @ X_c
    
    # 构造Q
    Q = -X_c.T @ Y
    
    # 构造G
    # part I: c_j >= 0 => - c_j <= 0
    G_11 = np.zeros((p, p))
    G_12 = -np.identity(p)
    G_1 = np.hstack((G_11, G_12))
    # part II: x_j <= c_j => x_j - c_j <= 0 
    G_21 = np.identity(p)
    G_22 = -np.identity(p)
    G_2 = np.hstack((G_21, G_22))
    # part III: -x_j <= c_j => -x_j - c_j <= 0
    G_31 = -np.identity(p)
    G_32 = -np.identity(p)
    G_3 = np.hstack((G_31, G_32))
    # part IV: c_1 + c_2 +...+ c_p <= t
    G_41 = np.zeros((1, p))
    G_42 = np.ones((1, p))
    G_4 = np.hstack((G_41, G_42))
    # G
    G = np.vstack((G_1, G_2, G_3, G_4))
    
    # 构造h
    h = np.zeros((3 * p + 1, 1 ))
    h[-1] = t
    
    res = solvers.qp(matrix(P), matrix(Q), matrix(G), matrix(h))
    return res['x'][:p]
# 2. Try LQA method to solve lasso problem, write your code to conduct simulation. 

# Analyze the stock return data, test the result of different variable selection methods for index tracking.