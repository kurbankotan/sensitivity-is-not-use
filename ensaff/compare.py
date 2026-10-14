r"""Paired comparison of two runs on the complexes they BOTH scored.

Structure mode drops complexes whose files are missing (n=4222 of 4250 in the first real run), so reading its
RMSE next to the sequence-mode or baseline RMSE compares different sets of molecules. This restricts every
comparison to the intersection of ids and reports a paired bootstrap interval for the difference, which also
accounts for the fact that the two models see the same test items.

    python -m ensaff.compare --a runs/struct_ens --b runs/seq_ens --out runs/compare
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def _metrics(y, p):
    return {"rmse": float(np.sqrt(((p - y) ** 2).mean())),
            "pearson": float(stats.pearsonr(y, p)[0]),
            "spearman": float(stats.spearmanr(y, p)[0])}


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=0):
    """95% interval for (B - A) on each metric, resampling complexes (both models share the resample)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    keys = ("rmse", "pearson", "spearman")
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 3:
            continue
        ma, mb = _metrics(y[idx], pa[idx]), _metrics(y[idx], pb[idx])
        for k in keys:
            diffs[k].append(mb[k] - ma[k])
    out = {}
    for k in keys:
        d = np.array(diffs[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                  "significant": bool(lo > 0 or hi < 0)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="reference run dir (e.g. runs/seq_ens)")
    p.add_argument("--b", required=True, help="run dir to compare against it (e.g. runs/struct_ens)")
    p.add_argument("--out", default="runs/compare")
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"config": {"a": a.a, "b": a.b, "n_boot": a.n_boot}}
    print(f"[compare] A={a.a}  B={a.b}  (fark = B - A, ortak alt kumede)")
    for fa in sorted(glob.glob(os.path.join(a.a, "pred_*.csv"))):
        split = os.path.basename(fa)[len("pred_"):-len(".csv")]
        fb = os.path.join(a.b, f"pred_{split}.csv")
        if not os.path.exists(fb):
            continue
        da = pd.read_csv(fa).set_index("id").sort_index()
        db = pd.read_csv(fb).set_index("id").sort_index()
        common = da.index.intersection(db.index)
        if len(common) < 20:
            continue
        da, db = da.loc[common], db.loc[common]
        y = da["y"].to_numpy(float)
        pa, pb = da["pred"].to_numpy(float), db["pred"].to_numpy(float)
        entry = {"n_common": int(len(common)), "n_a": int(len(pd.read_csv(fa))), "n_b": int(len(pd.read_csv(fb))),
                 "A": _metrics(y, pa), "B": _metrics(y, pb), "delta": paired_bootstrap(y, pa, pb, a.n_boot)}
        res[split] = entry
        print(f"  {split:<10} ortak n={entry['n_common']} (A:{entry['n_a']} B:{entry['n_b']})")
        for k in ("rmse", "pearson", "spearman"):
            d = entry["delta"][k]
            mark = "ANLAMLI" if d["significant"] else "fark yok"
            print(f"      {k:<9} A={entry['A'][k]:+.3f}  B={entry['B'][k]:+.3f}  "
                  f"fark={d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  {mark}")
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
