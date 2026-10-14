r"""Applicability domain: how far is each evaluation complex from the training data, and does that predict
the error better than the model's own uncertainty?

Motivation. On the leak-proof test split the learned variance head gives a weak positive signal, but on the
external BDB2020+ set it goes NEGATIVE: the model is confident exactly where it is wrong. Volkov et al.
(2022) argued that such models largely memorise, so proximity to the training set should govern accuracy.
This module measures that directly with two cheap distances:

  * ligand  - 1 - max Tanimoto (ECFP4) to any training ligand
  * protein - 1 - max 3-mer Jaccard to any training sequence

Both are computed in bulk (RDKit BulkTanimoto; sparse matrix product for the k-mer Jaccard) so they scale to
the full test split. The module reports, per split, the rank correlation of each signal with |error|, and
recalibrates conformal intervals using whichever signal is chosen.

    python -m ensaff.applicability --runs 'runs/seq_s*' --out runs/applicability
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .data import load_external_sets, load_lp_pdbbind
from .evaluate import conformal_quantile, regression_metrics, uncertainty_metrics

K = 3


def _fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None)
    return out


def max_tanimoto(query_smiles, ref_fps):
    from rdkit import DataStructs
    out = np.full(len(query_smiles), np.nan)
    for i, f in enumerate(_fps(query_smiles)):
        if f is not None and ref_fps:
            out[i] = max(DataStructs.BulkTanimotoSimilarity(f, ref_fps))
    return out


def _kmer_matrix(seqs, vocab):
    rows, cols = [], []
    for i, s in enumerate(seqs):
        s = str(s).replace(":", "")
        seen = {s[j:j + K] for j in range(len(s) - K + 1)}
        for km in seen:
            c = vocab.get(km)
            if c is not None:
                rows.append(i)
                cols.append(c)
    m = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(seqs), len(vocab)))
    return m


def max_kmer_jaccard(query_seqs, ref_seqs, block: int = 512):
    """Max 3-mer Jaccard of each query against any reference, via sparse intersection counts."""
    vocab = {}
    for s in list(ref_seqs) + list(query_seqs):
        s = str(s).replace(":", "")
        for j in range(len(s) - K + 1):
            vocab.setdefault(s[j:j + K], len(vocab))
    R = _kmer_matrix(ref_seqs, vocab)
    Q = _kmer_matrix(query_seqs, vocab)
    rn = np.asarray(R.sum(1)).ravel()
    qn = np.asarray(Q.sum(1)).ravel()
    out = np.zeros(Q.shape[0], dtype=np.float32)
    for i in range(0, Q.shape[0], block):
        q = Q[i:i + block]
        inter = np.asarray((q @ R.T).todense(), dtype=np.float32)
        union = qn[i:i + block, None] + rn[None, :] - inter
        out[i:i + block] = np.nanmax(np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0), axis=1)
    return out


def signals_for(pred_df, meta_df, ref_fps, ref_seqs):
    """Returns the per-row distances for one evaluation split, aligned to pred_df['id']."""
    meta = meta_df.set_index("id")
    smi = meta["smiles"].reindex(pred_df["id"]).values
    seq = meta["seq"].reindex(pred_df["id"]).values
    lig = 1.0 - max_tanimoto(smi, ref_fps)
    prot = 1.0 - max_kmer_jaccard(seq, ref_seqs)
    return lig, prot


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/seq_ens", help="run dir (or glob) holding pred_*.csv")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--out", default="runs/applicability")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--alpha", type=float, default=0.1)
    a = p.parse_args(argv)
    run_dirs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d))
    if not run_dirs:
        print(f"[applicability] kosu bulunamadi: {a.runs}")
        return None
    run = run_dirs[0]
    os.makedirs(a.out, exist_ok=True)

    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=a.clean_level)
    ref_fps = [f for f in _fps(tabs["train"].smiles) if f is not None]
    ref_seqs = list(pd.unique(tabs["train"].seq))
    meta = {"val": tabs["val"], "test": tabs["test"]}
    for name, (df, _pp, _lp) in load_external_sets(a.data_dir).items():
        meta[name] = df
    print(f"[applicability] referans: {len(ref_fps)} ligand, {len(ref_seqs)} protein | kosu: {run}")

    res = {"config": {"run": run, "k": K, "ref_ligands": len(ref_fps), "ref_proteins": len(ref_seqs)}}
    cal = None
    for f in sorted(glob.glob(os.path.join(run, "pred_*.csv"))):
        split = os.path.basename(f)[len("pred_"):-len(".csv")]
        if split not in meta:
            continue
        p_df = pd.read_csv(f)
        lig, prot = signals_for(p_df, meta[split], ref_fps, ref_seqs)
        err = np.abs(p_df["pred"] - p_df["y"]).values
        sd = p_df["sd"].values
        ok = ~(np.isnan(lig) | np.isnan(prot))
        rho = lambda x: float(stats.spearmanr(x[ok], err[ok])[0])  # noqa: E731
        both = lig * prot
        entry = {"n": int(ok.sum()),
                 "mean_lig_dist": float(np.nanmean(lig)), "mean_prot_dist": float(np.nanmean(prot)),
                 "rho_model_sd": rho(sd), "rho_lig_dist": rho(lig), "rho_prot_dist": rho(prot),
                 "rho_both_dist": rho(both)}
        # error of the closest vs furthest quartile, by the combined distance
        q = np.nanquantile(both[ok], [0.25, 0.75])
        near, far = err[ok][both[ok] <= q[0]], err[ok][both[ok] >= q[1]]
        entry["rmse_nearest_quartile"] = float(np.sqrt((near ** 2).mean())) if len(near) else None
        entry["rmse_furthest_quartile"] = float(np.sqrt((far ** 2).mean())) if len(far) else None
        # conformal intervals driven by the distance signal instead of the model's sd
        score = 0.25 + both                      # strictly positive width multiplier
        if split == "val":
            cal = conformal_quantile(p_df["y"].values[ok], p_df["pred"].values[ok], score[ok], a.alpha)
            res["conformal_q_distance"] = cal
        if cal is not None:
            lo, hi = p_df["pred"].values - cal * score, p_df["pred"].values + cal * score
            y = p_df["y"].values
            entry["coverage_distance"] = float(((y >= lo) & (y <= hi))[ok].mean())
            entry["width_distance"] = float((hi - lo)[ok].mean())
        res[split] = entry
        print(f"  {split:<10} n={entry['n']:<5} sd:{entry['rho_model_sd']:+.3f}  "
              f"ligand:{entry['rho_lig_dist']:+.3f}  protein:{entry['rho_prot_dist']:+.3f}  "
              f"ikisi:{entry['rho_both_dist']:+.3f}")
        pd.DataFrame({"id": p_df["id"], "lig_dist": lig, "prot_dist": prot, "abs_err": err, "sd": sd}) \
            .to_csv(os.path.join(a.out, f"signals_{split}.csv"), index=False)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
