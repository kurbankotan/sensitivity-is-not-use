"""Baseline: ECFP4 + mean protein embedding -> RandomForest (+ XGBoost if installed). Same splits as EnsAff.

    python -m ensaff.baseline --esm esm --out runs/baseline
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .data import load_external_sets, load_lp_pdbbind, split_chains
from .embed import ProteinEmbedder
from .evaluate import regression_metrics
from .features import ecfp


def featurize(df, emb):
    chains = [split_chains(s) for s in df.seq]
    emb.embed_chains([c for cs in chains for c in cs])
    P = np.stack([np.concatenate([emb.get(c) for c in cs], 0).astype(np.float32).mean(0) for cs in chains])
    L = np.stack([ecfp(s) for s in df.smiles]).astype(np.float32)
    return np.concatenate([L, P], 1), df.y.values.astype(float)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--esm", default="esm", choices=["esm", "onehot"])
    p.add_argument("--esm_model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--cache_dir", default="", help="where to keep the ESM embedding cache (default: data_dir)")
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="random_state for RF/XGB; vary it for error bars")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.cache_dir or a.data_dir,
                         f"emb_{a.esm}_{a.esm_model.split('/')[-1] if a.esm == 'esm' else 'oh'}.pt")
    emb = ProteinEmbedder(a.esm, a.esm_model, cache_path=cache)
    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"))
    for _n, (_df, _pp, _lp) in load_external_sets(a.data_dir).items():
        tabs[_n] = _df
    if a.limit:
        tabs["train"] = tabs["train"].sample(min(a.limit, len(tabs["train"])), random_state=0)
    X, y = featurize(tabs["train"], emb)
    models = {"rf": RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs=-1, random_state=a.seed)}
    try:
        from xgboost import XGBRegressor
        models["xgb"] = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=8, subsample=0.8,
                                     colsample_bytree=0.3, n_jobs=-1, random_state=a.seed)
    except ImportError:
        pass
    res = {"config": {"esm": a.esm, "esm_model": a.esm_model if a.esm == "esm" else "onehot",
                      "features": "ECFP4(2048) + mean ESM", "n_train": int(len(tabs["train"])), "seed": a.seed}}
    best_name = "xgb" if "xgb" in models else "rf"   # predictions saved for the stronger baseline
    feats = {}
    for split in [k for k in tabs if k != "train"]:
        if len(tabs[split]):
            feats[split] = featurize(tabs[split], emb)
    for name, m in models.items():
        m.fit(X, y)
        res[name] = {}
        for split, (Xs, ys) in feats.items():
            pred = m.predict(Xs)
            res[name][split] = regression_metrics(ys, pred)
            if name == best_name:
                # the stronger baseline's predictions are written out so ensaff.compare can run paired,
                # common-subset comparisons against it (sd is NaN: tree baselines give no uncertainty)
                pd.DataFrame({"id": tabs[split]["id"].values, "y": ys, "pred": pred, "sd": np.nan}) \
                    .to_csv(f"{a.out}/pred_{split}.csv", index=False)
    with open(f"{a.out}/metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
