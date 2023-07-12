"""
Code for the AppliedPhD project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import numpy as np
import pandas as pd

def extract_cb(CB):
    if len(CB.shape) == 2:
        (CB_00, CB_01), (CB_10, CB_11) = CB
    else:
        CB_00 = CB[:, 0, 0]
        CB_01 = CB[:, 0, 1]
        CB_10 = CB[:, 1, 0]
        CB_11 = CB[:, 1, 1]
    return CB_00, CB_01, CB_10, CB_11

def uplift_curve(y, t, score, use_churn_convention=True):
    if use_churn_convention:
        return profit_curve(y, t, score, np.array([[1, 1], [0, 0]]))
    else:
        return profit_curve(y, t, score, np.array([[0, 0], [1, 1]]))

def profit_curve(y, t, score, CB=np.array([[1, 1], [0, 0]])):
    score_ranking = np.argsort(score)[::-1]
    y = y[score_ranking]
    t = t[score_ranking]
    N = y.shape[0]
    score = score[score_ranking]
    CB_00, CB_01, CB_10, CB_11 = extract_cb(CB)

    grid = pd.DataFrame(data={
        "k": np.arange(N),
        "n_1": np.cumsum(t),
        "n_0": np.cumsum(1 - t),
        "score": score,
        "r_00": np.cumsum((1 - y) * (1 - t) * CB_00),
        "r_01": np.cumsum((1 - y) * t * CB_01),
        "r_10": np.cumsum(y * (1 - t) * CB_10),
        "r_11": np.cumsum(y * t * CB_11)
    })
    # Avoid NaNs, they will be replaced by zeros
    grid.loc[grid["n_0"] == 0, ["n_0"]] = 1
    grid.loc[grid["n_1"] == 0, ["n_1"]] = 1

    grid["profit"] = (
        ((grid.r_01 + grid.r_11) / grid.n_1) -
        ((grid.r_00 + grid.r_10) / grid.n_0)
    ) * grid["k"]

    return grid

def cf_profit_curve(score, S_0, S_1, CB=np.array([[1, 1], [0, 0]])):
    score_ranking = np.argsort(score)[::-1]
    score = score[score_ranking]
    N = score.shape[0]
    S_0 = S_0[score_ranking]
    S_1 = S_1[score_ranking]
    CB_00, CB_01, CB_10, CB_11 = extract_cb(CB)
    
    grid = pd.DataFrame(data={
        "k": np.arange(N),
        "score": score
    })
    grid["profit"] = np.cumsum(
        CB_01 * (1 - S_1) + CB_11 * S_1
        - CB_00 * (1 - S_0) - CB_10 * (1 - S_0)
    ) / N
    return grid

def calibrate_score(score, p):
    """
    Calibration of posterior probabilities as shown in
    Dal Pozzolo, Andrea, et al. "Calibrating probability
    with undersampling for unbalanced classification."
    2015 IEEE Symposium Series on Computational Intelligence.
    IEEE, 2015.
    """
    p_s = np.mean(score)
    ratio = p * (p_s - 1) / (p_s * (p - 1))
    return ratio * score / (ratio * score - score + 1)

