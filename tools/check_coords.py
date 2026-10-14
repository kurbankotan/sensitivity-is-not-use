"""Atom sayisi degil, KOORDINATLAR da kuratorlu ligandla ayni mi?"""
import os, sys, glob
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "BDB2020+.csv"))
good = bad = skip = 0; worst = []
for r in df.itertuples():
    mine = f"/tmp/extract_test/{r.pdbid}/ligand.pdb"
    ref_f = os.path.join(ROOT, "dataset", r.pdbid, "ligand.sdf")
    if not (os.path.exists(mine) and os.path.exists(ref_f)): skip += 1; continue
    ms = [m for m in Chem.SDMolSupplier(ref_f, sanitize=False) if m is not None]
    if not ms: skip += 1; continue
    ref = Chem.RemoveHs(ms[0], sanitize=False)
    R = ref.GetConformer().GetPositions()
    M = []
    for line in open(mine):
        if line.startswith("HETATM"):
            el = line[76:78].strip()
            if el.upper() == "H": continue
            M.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    M = np.array(M)
    if M.shape != R.shape: bad += 1; worst.append((r.pdbid, "sekil", M.shape, R.shape)); continue
    # siralamadan bagimsiz karsilastirma: her ref atomuna en yakin cikarilan atom
    d = np.sqrt(((R[:, None, :] - M[None, :, :]) ** 2).sum(-1)).min(1)
    if d.max() < 0.01: good += 1
    else: bad += 1; worst.append((r.pdbid, "mesafe", round(float(d.max()), 3), None))
print(f"koordinat kontrolu: tam ayni {good} | farkli {bad} | atlanan {skip}")
for w in worst[:8]: print("  ", w)
