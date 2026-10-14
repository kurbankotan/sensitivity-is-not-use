r"""Contact-count baseline: the POSITIVE CONTROL for the perturbation probes.

The probes in `geom_probe` report that damaging the residue-atom distance channel barely moves the
pair-representation model's predictions. On its own that is ambiguous: it could mean the model ignores the
channel, or it could mean the probes are too weak to move any model. This module settles it.

The features here are counts of (pocket residue class, ligand atom class) pairs falling in each distance
shell, in the spirit of RF-Score (Ballester & Mitchell, 2010). Two properties make it the control this study
needs:

  * It is built from EXACTLY the same inputs the neural model receives (the same pocket residues, the same
    residue-atom distance matrix, the same ligand). Any difference in behaviour is the model, not the input.
  * Distance is the whole feature set. Collapse every distance to a constant and the shells merge into one,
    leaving only composition. A model that uses geometry MUST lose accuracy under that damage.

Two of the probes damage this model in different ways, and both are reported. `permute_res` keeps the
multiset of distances and only breaks which residue each distance belongs to, so the number of counted
contacts is unchanged and the damage is matched exactly to what the same probe does to the neural model.
`const:8` is stronger: with no distances there is no distance cutoff either, so every residue-atom pair in
the pocket is counted once and the shell axis disappears. That is the honest meaning of removing geometry
(a cutoff is itself geometric information), but it changes the contact total as well as its resolution, so
`permute_res` is the cleaner like-for-like comparison and `const:8` the upper bound on the damage.

So if `const:8` destroys this model while leaving the neural model untouched, the information is present in
the channel, the probe can detect its removal, and the neural model is not using it. If it destroys both,
the probe is too blunt and the neural result means nothing.

    python -m ensaff.contactcount --probes none,translate:2.0,permute_res,const:8 --out runs/cc

CPU only; no GPU and no high RAM needed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import StructAffinityDataset, load_external_sets, load_lp_pdbbind
from .evaluate import regression_metrics
from .features import ELEMENTS

# Residues grouped by the property that drives the contact, not by identity: 20 classes against 5 distance
# shells and 5 ligand classes would give 500 sparse features on 7393 training complexes.
RES_CLASS = {"A": 0, "V": 0, "L": 0, "I": 0, "M": 0, "P": 0, "G": 0,          # hydrophobic/small
             "F": 1, "W": 1, "Y": 1,                                          # aromatic
             "S": 2, "T": 2, "N": 2, "Q": 2, "C": 2,                          # polar
             "D": 3, "E": 3,                                                  # acidic
             "K": 4, "R": 4, "H": 4}                                          # basic
N_RES_CLASS = 5
LIG_CLASS = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 3, "F": 4, "Cl": 4, "Br": 4, "I": 4, "B": 4}
N_LIG_CLASS = 5
SHELL_EDGES = (4.0, 6.0, 8.0, 10.0, 12.0)      # last edge doubles as the cutoff
N_SHELL = len(SHELL_EDGES)
N_FEAT = N_RES_CLASS * N_LIG_CLASS * N_SHELL


def _lig_classes(x: np.ndarray) -> np.ndarray:
    """Element class per ligand heavy atom, read back from the one-hot block of the atom features."""
    idx = x[:, :len(ELEMENTS) + 1].argmax(1)
    out = np.full(len(idx), N_LIG_CLASS - 1, dtype=np.int64)
    for i, e in enumerate(ELEMENTS):
        out[idx == i] = LIG_CLASS.get(e, N_LIG_CLASS - 1)
    return out


def featurize_item(item, res_letters) -> np.ndarray:
    """item: a StructAffinityDataset sample (its `dist` already carries any geom/probe damage)."""
    D = np.asarray(item["dist"])[0]                       # [residues, ligand atoms], first state
    rc = np.array([RES_CLASS.get(c, 0) for c in res_letters], dtype=np.int64)
    lc = _lig_classes(np.asarray(item["x"]))
    shell = np.searchsorted(np.asarray(SHELL_EDGES), D, side="left")   # 0..N_SHELL, N_SHELL = beyond cutoff
    feat = np.zeros((N_RES_CLASS, N_LIG_CLASS, N_SHELL), dtype=np.float32)
    keep = shell < N_SHELL
    if keep.any():
        ri, ai = np.nonzero(keep)
        np.add.at(feat, (rc[ri], lc[ai], shell[ri, ai]), 1.0)
    return feat.reshape(-1)


def featurize_dataset(ds: StructAffinityDataset):
    X, y, ids = [], [], []
    for i in range(len(ds)):
        rec = ds.recs[i]
        letters = [rec["chain_seqs"][ci][ri] for ci, ri in rec["pocket"]]
        X.append(featurize_item(ds[i], letters))
        y.append(rec["y"])
        ids.append(rec["id"])
    return np.stack(X) if X else np.zeros((0, N_FEAT), np.float32), np.asarray(y, float), ids


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cc")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--protein_pattern", default="pdbbind/{id}/{id}_protein.pdb")
    p.add_argument("--ligand_pattern", default="pdbbind/{id}/{id}_ligand.sdf")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--max_res", type=int, default=64)
    p.add_argument("--max_atoms", type=int, default=72)
    p.add_argument("--max_poses", type=int, default=4)
    p.add_argument("--probes", default="none,translate:1.0,translate:4.0,rotate,permute_res,const:8")
    p.add_argument("--splits", default="test,BDB2020+")
    p.add_argument("--seeds", default="0,1,2,3,4")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    class _NoEmb:              # the contact-count model needs no protein embeddings
        dim = 1
        def embed_chains(self, seqs): pass
        def get(self, seq): return np.zeros((len(seq), 1), dtype=np.float32)

    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=a.clean_level)
    exts = load_external_sets(a.data_dir)
    mk = lambda df, n, pp, lp: StructAffinityDataset(  # noqa: E731
        df, _NoEmb(), pp, lp, a.cutoff, a.max_res, a.max_atoms, a.max_poses, 0, 0.5, n,
        eval_states=0, keep_coords=True)
    dsets = {"train": mk(tabs["train"], "cc-train", a.protein_pattern, a.ligand_pattern),
             "val": mk(tabs["val"], "cc-val", a.protein_pattern, a.ligand_pattern),
             "test": mk(tabs["test"], "cc-test", a.protein_pattern, a.ligand_pattern)}
    for name, (df, pp, lp) in exts.items():
        dsets[name] = mk(df, f"cc-{name}", pp, lp)

    Xtr, ytr, _ = featurize_dataset(dsets["train"])
    print(f"[cc] egitim ozellik matrisi {Xtr.shape}, ortalama temas {Xtr.sum(1).mean():.1f}")

    from sklearn.ensemble import RandomForestRegressor
    try:
        from xgboost import XGBRegressor
        have_xgb = True
    except Exception as e:
        print("[cc] xgboost yok, yalnizca random forest:", e)
        have_xgb = False

    want = [s.strip() for s in a.splits.split(",") if s.strip()]
    probes = [s.strip() for s in a.probes.split(",") if s.strip()]
    if probes[0] != "none":
        probes = ["none"] + [x for x in probes if x != "none"]
    seeds = [int(s) for s in a.seeds.split(",")]

    res = {"config": {"features": f"{N_RES_CLASS}x{N_LIG_CLASS}x{N_SHELL} temas sayimi = {N_FEAT}",
                      "shells": list(SHELL_EDGES), "n_train": int(len(ytr)), "seeds": seeds,
                      "probes": probes, "note": "pozitif kontrol: geometri butun ozellik kumesi"}}
    rows = []
    for sd in seeds:
        models = {"rf": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=sd)}
        if have_xgb:
            models["xgb"] = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, n_jobs=-1, random_state=sd)
        for mn, m in models.items():
            m.fit(Xtr, ytr)
            for probe in probes:
                for split in want:
                    ds = dsets.get(split)
                    if ds is None or not len(ds):
                        continue
                    ds.probe = probe
                    Xs, ys, _ids = featurize_dataset(ds)
                    ds.probe = "none"
                    mt = regression_metrics(ys, m.predict(Xs))
                    res.setdefault(f"{mn}:{probe}", {})[split] = mt
                    rows.append((mn, probe, split, sd, mt))
                    print(f"  seed {sd} {mn:4s} {probe:14s} {split:10s} n={mt['n']:<5} "
                          f"rmse={mt['rmse']:.3f} r={mt['pearson']:.3f}", flush=True)

    lines = ["(POZITIF KONTROL: ozellikler tamamen mesafeden tureyen temas sayimlari)",
             "(const:8 ya da permute_res bu modeli bozmazsa sonda yontemi gecersizdir)",
             "(not: mesafe silinince mesafe esigi de uygulanamaz, bu yuzden const:8 altinda cepteki her",
             " kalinti-atom cifti bir kez sayilir; permute_res ise temas toplamini aynen korur ve sinir",
             " agina uygulanan hasarla birebir eslesir)", ""]
    for mn in sorted({r[0] for r in rows}):
        for split in want:
            got0 = [r[4]["rmse"] for r in rows if r[0] == mn and r[1] == "none" and r[2] == split]
            if not got0:
                continue
            base = float(np.mean(got0))
            lines.append(f"  {mn} / {split}  (bozulmamis rmse = {base:.3f} pK, {len(got0)} tohum)")
            for probe in probes:
                g = [r[4]["rmse"] for r in rows if r[0] == mn and r[1] == probe and r[2] == split]
                if not g:
                    continue
                m_, s_ = float(np.mean(g)), float(np.std(g, ddof=1)) if len(g) > 1 else 0.0
                lines.append(f"    {probe:16s} rmse={m_:7.3f}+-{s_:<5.3f}  bedel={m_ - base:+.3f} pK")
            lines.append("")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(a.out, "probe_report.txt"), "w") as fh:
        fh.write(txt)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    pd.DataFrame([{"model": r[0], "probe": r[1], "split": r[2], "seed": r[3], **r[4]} for r in rows]) \
        .to_csv(os.path.join(a.out, "probe_rows.csv"), index=False)
    return res


if __name__ == "__main__":
    main()
