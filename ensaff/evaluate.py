"""Metrics, split-conformal calibration and uncertainty diagnostics."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def regression_metrics(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    err = mu - y
    return {"n": int(len(y)), "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "pearson": float(stats.pearsonr(y, mu)[0]) if len(y) > 2 else float("nan"),
            "spearman": float(stats.spearmanr(y, mu)[0]) if len(y) > 2 else float("nan")}


def conformal_quantile(y_cal, mu_cal, sd_cal, alpha=0.1):
    """Split-conformal with normalised residuals |y-mu|/sd -> interval mu +- q*sd has >= 1-alpha coverage.

    The guarantee assumes the calibration residuals are exchangeable with test AND independent of model
    selection. Using the early-stopping split for calibration breaks that independence and the intervals
    under-cover, so train.py holds out a separate calibration half.
    """
    s = np.abs(np.asarray(y_cal) - np.asarray(mu_cal)) / np.asarray(sd_cal)
    n = len(s)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def uncertainty_metrics(y, mu, sd, q=None, alpha=0.1):
    y, mu, sd = map(lambda a: np.asarray(a, float), (y, mu, sd))
    err = np.abs(mu - y)
    out = {"nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + (y - mu) ** 2 / (2 * sd ** 2))),
           "spearman_sd_vs_err": float(stats.spearmanr(sd, err)[0]) if len(y) > 2 else float("nan")}
    order = np.argsort(sd)
    for frac in (1.0, 0.75, 0.5, 0.25):
        k = max(1, int(round(frac * len(y))))
        idx = order[:k]
        out[f"rmse_top{int(frac * 100)}%conf"] = float(np.sqrt(((mu[idx] - y[idx]) ** 2).mean()))
    if q is not None and np.isfinite(q):
        lo, hi = mu - q * sd, mu + q * sd
        out[f"coverage@{int((1 - alpha) * 100)}"] = float(((y >= lo) & (y <= hi)).mean())
        out["mean_interval_width"] = float((hi - lo).mean())
    return out
