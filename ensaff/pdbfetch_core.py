"""Extract (protein, ligand) from a raw RCSB PDB entry, using the expected ligand from SMILES.

Why this exists: PDBbind requires registration, but the same structures are freely available from RCSB by
PDB ID. The curation PDBbind adds is mainly "which HETATM group is the actual ligand" - crystallisation
additives, ions and waters must be rejected. Here that choice is made by matching the heavy-atom count of
each candidate group against the heavy-atom count of the SMILES recorded in the LP-PDBBind metadata.
"""
from __future__ import annotations

import os
import sys

# waters, ions and the usual crystallisation additives: never the ligand of interest
JUNK = {
    "HOH", "DOD", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "MPD", "ACT", "ACY",
    "FMT", "DMS", "TRS", "EPE", "MES", "IMD", "BME", "NO3", "CIT", "FLC", "TLA", "IPA", "MOH", "ETA", "BU3",
    "NH4", "AZI", "SCN", "CO3", "BCT", "OXL", "MLI", "SIN", "TAR", "BEN", "CAC", "NHE", "P6G", "12P", "15P",
    "NA", "K", "MG", "ZN", "CA", "MN", "FE", "FE2", "NI", "CU", "CU1", "CD", "CO", "HG", "CL", "BR", "IOD",
    "F", "LI", "RB", "CS", "SR", "BA", "AU", "AG", "PT", "PB", "XE", "KR", "AR", "UNX", "UNL",
}


def heavy_atoms_from_smiles(smiles: str) -> int | None:
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(str(smiles))
        return None if m is None else Chem.RemoveHs(m, sanitize=False).GetNumAtoms()
    except Exception:
        return None


def parse_entry(path: str):
    """Returns (protein_lines, {(resname, chain, resseq): [hetatm lines]}) from the first model."""
    prot, het = [], {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line[:6] in ("ATOM  ", "HETATM") and line[16] not in (" ", "A", "1"):
                continue                 # alternate conformation: keep only the first
            if line.startswith("ATOM"):
                prot.append(line)
            elif line.startswith("HETATM"):
                resn = line[17:20].strip()
                if resn in ("HOH", "DOD", "WAT"):
                    continue
                if resn == "MSE":          # selenomethionine is part of the protein chain
                    prot.append(line)
                    continue
                het.setdefault((resn, line[21], line[22:27]), []).append(line)
    return prot, het


def _is_hydrogen(line: str) -> bool:
    el = line[76:78].strip()
    return el.upper() == "H" if el else line[12:16].strip().startswith("H")


def _xyz(lines):
    import numpy as np
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                     for l in lines if not _is_hydrogen(l)], dtype=float)


def _contacts(lig_lines, prot_lines, cutoff=5.0):
    """How many protein atoms sit within `cutoff` of this ligand copy - a proxy for how well the
    binding site is defined. Used to break ties when a structure holds several copies of the ligand."""
    import numpy as np
    L = _xyz(lig_lines)
    if len(L) == 0 or not prot_lines:
        return 0
    P = _xyz(prot_lines)
    if len(P) == 0:
        return 0
    lo, hi = L.min(0) - cutoff, L.max(0) + cutoff          # cheap bounding-box prefilter
    P = P[((P >= lo) & (P <= hi)).all(1)]
    if len(P) == 0:
        return 0
    d = np.sqrt(((P[:, None, :] - L[None, :, :]) ** 2).sum(-1))
    return int((d.min(1) < cutoff).sum())


def pick_ligand(het: dict, want: int | None, prot_lines=None):
    """Choose the HETATM group that best matches the expected heavy-atom count; among equally good
    matches (several copies of the same ligand) prefer the copy with the most protein contacts."""
    cands = []
    for key, lines in het.items():
        if key[0] in JUNK:
            continue
        n = sum(1 for l in lines if not _is_hydrogen(l))
        if n < 5:                       # too small to be a drug-like ligand
            continue
        cands.append((key, lines, n))
    if not cands:
        return None, None, "aday yok"
    if want:
        best = min(abs(c[2] - want) for c in cands)
        tied = [c for c in cands if abs(c[2] - want) == best]
        # Several copies of the same ligand: take the first one in file order. A "most protein contacts"
        # tie-break was tried and matched the curated PDBbind choice far less often (78/114 vs 109/114),
        # so file order is what the curation effectively follows.
        key, lines, n = tied[0]
        how = ("tam eslesme" if n == want else f"yakin eslesme ({n} vs {want})") + \
              (f" [{len(tied)} kopya, ilki]" if len(tied) > 1 else "")
        return key, lines, how
    key, lines, n = max(cands, key=lambda c: c[2])
    return key, lines, "en buyuk grup (SMILES yok)"


def extract(entry_path: str, smiles: str | None, out_dir: str):
    want = heavy_atoms_from_smiles(smiles) if smiles else None
    prot, het = parse_entry(entry_path)
    key, lig_lines, how = pick_ligand(het, want, prot)
    if key is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "protein.pdb"), "w") as fh:
        fh.writelines(prot)
        fh.write("END\n")
    with open(os.path.join(out_dir, "ligand.pdb"), "w") as fh:
        fh.writelines(lig_lines)
        fh.write("END\n")
    return {"resname": key[0], "chain": key[1], "how": how,
            "n_heavy": sum(1 for l in lig_lines if not _is_hydrogen(l)), "want": want,
            "n_protein_atoms": len(prot)}


if __name__ == "__main__":
    print(extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3]))


def ligand_pdb_to_sdf(lig_pdb: str, smiles: str, out_sdf: str) -> str | None:
    """A PDB ligand carries no bond orders; recover them by matching against the SMILES template
    (RDKit AssignBondOrdersFromTemplate). Returns a short status string, or None on failure."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromPDBFile(lig_pdb, removeHs=True, sanitize=False)
    if mol is None:
        return None
    tpl = Chem.MolFromSmiles(str(smiles))
    if tpl is None:
        return None
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(tpl, mol)
        status = "sablonla eslesti"
    except Exception:
        try:                                   # fall back to the raw connectivity RDKit inferred
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                             catchErrors=True)
            fixed, status = mol, "sablon basarisiz, ham baglar"
        except Exception:
            return None
    w = Chem.SDWriter(out_sdf)
    w.write(fixed)
    w.close()
    return status
