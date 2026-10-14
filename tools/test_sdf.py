import os, sys, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_ligand import ligand_pdb_to_sdf
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "BDB2020+.csv"))
tpl_ok = raw = fail = skip = 0
for r in df.itertuples():
    lp = f"/tmp/extract_test/{r.pdbid}/ligand.pdb"
    if not os.path.exists(lp): skip += 1; continue
    st = ligand_pdb_to_sdf(lp, r.smiles, f"/tmp/extract_test/{r.pdbid}/ligand.sdf")
    if st is None: fail += 1
    elif "eslesti" in st: tpl_ok += 1
    else: raw += 1
print(f"SDF uretimi: sablonla tam {tpl_ok} | ham baglarla {raw} | basarisiz {fail} | atlanan {skip}")
# uretilen SDF'ler bizim featurizer'dan geciyor mu?
sys.path.insert(0, "/home/claude/ensaff")
from ensaff.features import load_ligand_poses, featurize_mol
good = 0; bad = 0
for r in df.itertuples():
    f = f"/tmp/extract_test/{r.pdbid}/ligand.sdf"
    if not os.path.exists(f): continue
    m, poses = load_ligand_poses(f)
    feat = featurize_mol(m) if m is not None else None
    if feat is not None and poses: good += 1
    else: bad += 1
print(f"featurizer'dan gecen: {good} | gecemeyen: {bad}")
