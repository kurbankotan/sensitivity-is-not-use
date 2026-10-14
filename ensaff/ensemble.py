r"""Deep-ensemble aggregation over the per-seed sequence-mode runs.

Averaging several independently seeded models and using their disagreement as uncertainty is the standard
deep-ensemble recipe (Lakshminarayanan et al., 2017); it is used here as an established baseline for
uncertainty, not as a contribution of this work. For a Gaussian mixture with equal weights the predictive
mean and variance are

    mu  = (1/M) * sum_i mu_i
    var = (1/M) * sum_i sigma_i^2   +   (1/M) * sum_i (mu_i - mu)^2
          \_____ aleatoric _____/       \_____ ensemble spread _____/

The conformal quantile is refitted on the validation split so coverage is calibrated for the ensemble.

    python -m ensaff.ensemble --runs 'runs/seq_s*' --out runs/seq_ens
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .evaluate import conformal_quantile, regression_metrics, uncertainty_metrics

BASE_SPLITS = ("val", "test")   # external sets are discovered from the files on disk
Q_DEGENERATE = 5.0   # above this the quantile, not the model, is setting the interval width


def _load(run_dirs, split):
    """Returns (ids, y, [mu per model], [sd per model]) aligned on id, or None if unavailable."""
    frames = []
    for d in run_dirs:
        p = os.path.join(d, f"pred_{split}.csv")
        if not os.path.exists(p):
            return None
        frames.append(pd.read_csv(p).set_index("id").sort_index())
    ids = frames[0].index
    for f in frames[1:]:
        if not f.index.equals(ids):          # align defensively; seeds must score the same complexes
            common = ids.intersection(f.index)
            if len(common) == 0:
                return None
            ids = common
    frames = [f.loc[ids] for f in frames]
    y = frames[0]["y"].to_numpy(float)
    mu = np.stack([f["pred"].to_numpy(float) for f in frames])
    sd = np.stack([f["sd"].to_numpy(float) for f in frames])
    return list(ids), y, mu, sd


def combine(mu, sd):
    """Mixture mean and variance. Tree baselines report no per-prediction sd (NaN), in which case the
    spread between members is the only uncertainty available - this keeps them ensemble-able too, so the
    comparison can be ensemble against ensemble rather than ensemble against a single model."""
    m = mu.mean(0)
    spread = ((mu - m) ** 2).mean(0)
    aleatoric = np.nan_to_num((sd ** 2), nan=0.0).mean(0)
    return m, np.sqrt(aleatoric + spread)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/seq_s*", help="glob of per-seed run directories")
    p.add_argument("--out", default="runs/seq_ens")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cal_seed", type=int, default=0, help="which half of val calibrates (fixed across runs)")
    a = p.parse_args(argv)
    run_dirs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d))
    if len(run_dirs) < 2:
        print(f"[ensemble] en az 2 kosu gerekli, bulunan: {run_dirs} -> atlaniyor")
        return None
    os.makedirs(a.out, exist_ok=True)
    print(f"[ensemble] {len(run_dirs)} kosu birlestiriliyor: {run_dirs}")

    cal = _load(run_dirs, "val")
    if cal is None:
        print("[ensemble] val tahminleri yok -> atlaniyor")
        return None
    _, yv, muv, sdv = cal
    mv, sv = combine(muv, sdv)
    # Every member was early-stopped on this same val split. Calibrating on all of it makes the residuals
    # dependent on model selection, so q comes out too tight and the nominal 90% intervals under-cover on
    # test (v11: val 0.902 against test 0.874). Calibrate on one fixed half instead, as train.py does.
    rng = np.random.RandomState(a.cal_seed)
    half = rng.rand(len(yv)) < 0.5
    if half.sum() < 50 or (~half).sum() < 50:
        half = np.ones(len(yv), dtype=bool)
        print("[ensemble] val cok kucuk -> ayri kalibrasyon yarisi kullanilamadi")
    q = conformal_quantile(yv[half], mv[half], sv[half], a.alpha) if np.all(sv[half] > 0) else float("nan")
    degenerate = bool(np.isfinite(q) and q > Q_DEGENERATE)
    if degenerate:
        print(f"  !! UYARI: konformal q={q:.2f} dejenere. Uyeler birbirine neredeyse hic karsi cikmiyor "
              f"(tahmin sd'si ~0), kapsama tamamen bu buyuk carpanla saglaniyor; bu toplulugun "
              f"belirsizlik sayilari (cov90, genislik) yorumlanamaz.")

    # whatever splits the member runs actually scored (val, test, BDB2020+, mpro, EGFR, ...)
    import re as _re
    splits = sorted({_re.sub(r"^pred_|\.csv$", "", os.path.basename(f))
                     for f in glob.glob(os.path.join(run_dirs[0], "pred_*.csv"))},
                    key=lambda x: (x not in BASE_SPLITS, x))
    res = {"config": {"members": len(run_dirs), "runs": run_dirs, "splits": splits,
                      "method": "deep ensemble (mean + mixture variance), conformal refit on val"},
           "conformal_q": q, "conformal_n_cal": int(half.sum()), "conformal_degenerate": degenerate,
           "conformal_note": "calibrated on a val half disjoint from early stopping"}
    for split in splits:
        got = _load(run_dirs, split)
        if got is None:
            continue
        ids, y, mu, sd = got
        m, s = combine(mu, sd)
        res[split] = {**regression_metrics(y, m), **uncertainty_metrics(y, m, s, q, a.alpha)}
        pd.DataFrame({"id": ids, "y": y, "pred": m, "sd": s, "lo": m - q * s, "hi": m + q * s}) \
            .to_csv(os.path.join(a.out, f"pred_{split}.csv"), index=False)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    for k in splits:
        if k in res:
            r = res[k]
            print(f"  {k:<12} rmse={r['rmse']:.3f} r={r['pearson']:.3f} rho={r['spearman']:.3f} "
                  f"cov90={r.get(f'coverage@{int((1-a.alpha)*100)}'):.3f} sd~err={r['spearman_sd_vs_err']:.3f}")
    return res


if __name__ == "__main__":
    main()
