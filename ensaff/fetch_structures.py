r"""Build the structure folder for LP-PDBBind from RCSB PDB - no PDBbind registration needed.

LP-PDBBind ships metadata only (sequence, SMILES, affinity); the 3D files normally come from PDBbind,
which requires registration. The same structures are openly available from RCSB by PDB ID, so this script
downloads them and reproduces the two files the structure-mode dataset expects:

    <out>/<id>/protein.pdb     standard-residue ATOM records (full chains, so ESM sees full context)
    <out>/<id>/ligand.sdf      the bound ligand, bond orders recovered from the LP-PDBBind SMILES

Which HETATM group is the ligand is decided by matching heavy-atom counts against that SMILES. On the 115
BDB2020+ complexes, where a curated ligand is available for comparison, this picked the correct ligand for
114/115 and reproduced the curated coordinates exactly for 109/115 (the other 5 are the same ligand in a
symmetry-equivalent copy). Bond-order assignment succeeded from the template for 105 and fell back to raw
connectivity for 9, with all 114 usable by the featurizer.

    python -m ensaff.fetch_structures --out /content/pdbstruct --workers 8
"""
from __future__ import annotations

import argparse
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .data import load_lp_pdbbind
from .pdbfetch_core import extract, ligand_pdb_to_sdf

RCSB = "https://files.rcsb.org/download/{id}.pdb"


def fetch_one(pdb_id: str, smiles: str, out_root: str, retries: int = 3, timeout: int = 30):
    pid = str(pdb_id).lower()
    d = os.path.join(out_root, pid)
    if os.path.exists(os.path.join(d, "ligand.sdf")) and os.path.exists(os.path.join(d, "protein.pdb")):
        return pid, "zaten var"
    raw = os.path.join(out_root, "_raw", f"{pid}.pdb")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    if not os.path.exists(raw):
        for k in range(retries):
            try:
                req = urllib.request.Request(RCSB.format(id=pid), headers={"User-Agent": "ensaff-research/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r, open(raw, "wb") as fh:
                    fh.write(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return pid, "RCSB'de yok (404)"
                time.sleep(1.5 * (k + 1) + random.random())
            except Exception:
                time.sleep(1.5 * (k + 1) + random.random())
        else:
            return pid, "indirilemedi"
    info = extract(raw, smiles, d)
    if info is None:
        return pid, "ligand bulunamadi"
    st = ligand_pdb_to_sdf(os.path.join(d, "ligand.pdb"), smiles, os.path.join(d, "ligand.sdf"))
    if st is None:
        return pid, "sdf uretilemedi"
    try:
        os.remove(raw)                       # raw entry is no longer needed; keeps the folder small
    except OSError:
        pass
    return pid, ("tamam" if "eslesti" in st else "tamam (ham baglar)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".")
    p.add_argument("--out", default="/content/pdbstruct")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--clean_level", default="CL1")
    a = p.parse_args(argv)
    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=a.clean_level)
    df = pd.concat([tabs[s] for s in ("train", "val", "test")], ignore_index=True)
    if a.limit:
        df = df.head(a.limit)
    os.makedirs(a.out, exist_ok=True)
    print(f"[fetch] {len(df)} kompleks -> {a.out}  ({a.workers} paralel indirme)")
    t0, done = time.time(), []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch_one, r.id, r.smiles, a.out) for r in df.itertuples()]
        for i, f in enumerate(futs):
            done.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
    counts = pd.Series([s for _, s in done]).value_counts()
    print("\n[fetch] sonuc:")
    for k, v in counts.items():
        print(f"  {k:<28} {v}")
    usable = sum(v for k, v in counts.items() if k.startswith("tamam") or k == "zaten var")
    print(f"\n[fetch] kullanilabilir: {usable}/{len(df)} (%{100 * usable / max(1, len(df)):.1f})")
    pd.DataFrame(done, columns=["id", "durum"]).to_csv(os.path.join(a.out, "_fetch_log.csv"), index=False)
    return counts


if __name__ == "__main__":
    main()
