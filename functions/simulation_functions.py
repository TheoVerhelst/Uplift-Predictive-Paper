"""
Code for the AppliedPhD project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import pandas as pd
from scipy.stats import binom, norm
import numpy as np

def simulate_uplift_norm(N_train, X, noise, lambda_0, lambda_1, eta_0, eta_1, p, C=10, random_state=None):
    # Split parameters and data between observed and unobserved components
    N = X.shape[0]
    Y_0 = np.dot(X, lambda_0) + noise > eta_0
    Y_1 = np.dot(X, lambda_1) + noise > eta_1
    T = st.uniform.rvs(size=N, random_state=random_state) <= p
    Y = Y_0
    Y[T] = Y_1[T]
    
    # Generate the train/test split
    X_train = X[:N_train]
    Y_train = Y[:N_train]
    T_train = T[:N_train]
    X_test = X[N_train:]
    Y_test = Y[N_train:]
    T_test = T[N_train:]
    
    S_0_test = st.norm.cdf(np.dot(X_test, lambda_0))
    S_1_test = st.norm.cdf(np.dot(X_test, lambda_1))
    
    model_0 = LogisticRegression(C=C)
    model_0.fit(X_train[~T_train, :], Y_train[~T_train])
    model_1 = LogisticRegression(C=C)
    model_1.fit(X_train[T_train, :], Y_train[T_train])
    
    S_0_hat_test = model_0.predict_proba(X_test)[:, 1]
    S_1_hat_test = model_1.predict_proba(X_test)[:, 1]
    
    uplift_hat_test = S_0_hat_test - S_1_hat_test
    CB = np.array([[1, 1], [0, 0]])
    
    curve_u = cf_profit_curve(uplift_hat_test, S_0_test, S_1_test, CB)
    curve_p = cf_profit_curve(S_0_hat_test, S_0_test, S_1_test, CB)
    auuc_u = np.mean(curve_u.profit)
    auuc_p = np.mean(curve_p.profit)
    return auuc_u, auuc_p

def simulate_uplift_dirichlet(a, size, n_p_0, n_u_0, n_u_1, use_churn_convention=True, random_state=None):
    rng = np.random.default_rng(seed=random_state)
    mu = rng.dirichlet(a, size=size)
    S_0 = mu[:, 1] + mu[:, 3]
    S_1 = mu[:, 2] + mu[:, 3]
    uplift = S_0 - S_1
    
    # Estimators
    S_0_hat = binom.rvs(n_p_0, S_0) / n_p_0
    S_0_u = binom.rvs(n_u_0, S_0) / n_u_0
    S_1_u = binom.rvs(n_u_1, S_1) / n_u_1
    uplift_hat = S_0_u - S_1_u
    
    if not use_churn_convention:
        uplift = -uplift
        uplift_hat = -uplift_hat
        
    return pd.DataFrame.from_dict({
        "alpha": mu[:, 0],
        "beta" : mu[:, 1],
        "gamma": mu[:, 2],
        "delta": mu[:, 3],
        "S_0": S_0,
        "S_1": S_1,
        "S_0_hat": S_0_hat,
        "S_0_u": S_0_u,
        "S_1_u": S_1_u,
        "uplift": uplift,
        "uplift_hat": uplift_hat
    })