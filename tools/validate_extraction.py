"""Validate automatic extraction against BDB2020+'s curated ligand.sdf (the ground truth)."""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_ligand import extract

ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "BDB2020+.csv"))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ok = exact = near = missing = mismatch = 0
rows = []
for r in df.itertuples():
    d = os.path.join(ROOT, "dataset", r.pdbid)
    ent = glob.glob(os.path.join(d, "*.ent"))
    ref_sdf = os.path.join(d, "ligand.sdf")
    if not ent or not os.path.exists(ref_sdf):
        missing += 1; continue
    ref = Chem.SDMolSupplier(ref_sdf, sanitize=False)
    ref = [m for m in ref if m is not None]
    if not ref:
        missing += 1; continue
    n_ref = Chem.RemoveHs(ref[0], sanitize=False).GetNumAtoms()
    out = os.path.join("/tmp/extract_test", r.pdbid)
    res = extract(ent[0], r.smiles, out)
    if res is None:
        mismatch += 1; rows.append((r.pdbid, "CIKARILAMADI", n_ref, None)); continue
    ok += 1
    if res["n_heavy"] == n_ref: exact += 1
    elif abs(res["n_heavy"] - n_ref) <= 2: near += 1
    else: mismatch += 1; rows.append((r.pdbid, res["resname"], n_ref, res["n_heavy"]))
print(f"toplam {len(df)} | cikarildi {ok} | eksik dosya {missing}")
print(f"  atom sayisi TAM eslesme : {exact}  (%{100*exact/max(1,len(df)):.1f})")
print(f"  +-2 atom farki          : {near}")
print(f"  uyusmayan               : {mismatch}")
if rows:
    print("\nuyusmayanlar (pdbid, secilen, beklenen_atom, secilen_atom):")
    for x in rows[:12]: print("  ", x)
