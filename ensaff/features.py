"""Featurization: ligands (RDKit), protein structures (PDB parsing, pocket extraction)."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------------- ligand
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
BOND_TYPES = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
N_BOND = 5          # 0 = no bond
MAX_TOPO = 8        # graph distances clipped to 8; 9 = disconnected
N_TOPO = MAX_TOPO + 2
LIG_FEAT_DIM = (len(ELEMENTS) + 1) + 6 + 3 + (len(HYBRID) + 1) + 1 + 1 + 4 + 1  # = 31


def _onehot(v, choices, other=True):
    out = [0.0] * (len(choices) + (1 if other else 0))
    if v in choices:
        out[choices.index(v)] = 1.0
    elif other:
        out[-1] = 1.0
    return out


def atom_features(a: Chem.Atom) -> list[float]:
    f = _onehot(a.GetSymbol(), ELEMENTS)                       # 11
    f += _onehot(min(a.GetDegree(), 5), [0, 1, 2, 3, 4, 5], other=False)  # 6
    f += _onehot(max(-1, min(1, a.GetFormalCharge())), [-1, 0, 1], other=False)  # 3
    f += _onehot(a.GetHybridization(), HYBRID)                 # 4
    f += [float(a.GetIsAromatic())]                            # 1
    f += [float(a.IsInRing())]                                 # 1
    f += _onehot(min(a.GetTotalNumHs(), 3), [0, 1, 2, 3], other=False)  # 4
    f += [float(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)]  # 1
    return f


def featurize_mol(mol: Chem.Mol, max_atoms: int = 72, min_atoms: int = 5):
    """Heavy-atom graph. Returns dict(x [N,F], bond [N,N] int8, topo [N,N] int8) or None."""
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol, sanitize=False)
    n = mol.GetNumAtoms()
    if n < min_atoms or n > max_atoms:
        return None
    x = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    bond = np.zeros((n, n), dtype=np.int8)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = BOND_TYPES.get(b.GetBondType(), 1)
        bond[i, j] = bond[j, i] = t
    topo = Chem.GetDistanceMatrix(mol)
    topo = np.where(topo > 1e6, MAX_TOPO + 1, np.minimum(topo, MAX_TOPO)).astype(np.int8)
    return {"x": x, "bond": bond, "topo": topo}


def featurize_smiles(smiles: str, max_atoms: int = 72):
    mol = Chem.MolFromSmiles(str(smiles))
    return featurize_mol(mol, max_atoms=max_atoms)


def ecfp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros(n_bits, dtype=np.uint8)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    on = list(fp.GetOnBits())
    arr[on] = 1
    return arr


def load_ligand_poses(sdf_path: str, max_poses: int = 8):
    """Read all poses from an SDF. Returns (mol_of_first_pose, [coords [N,3] heavy atoms]) or (None, [])."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols = [m for m in suppl if m is not None]
    if not mols:  # retry without sanitization
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        mols = []
        for m in suppl:
            if m is None:
                continue
            try:
                m.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                                 | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                                 catchErrors=True)
                mols.append(m)
            except Exception:
                continue
    if not mols:
        return None, []
    ref = Chem.RemoveHs(mols[0], sanitize=False)
    n = ref.GetNumAtoms()
    poses = []
    for m in mols[:max_poses]:
        m = Chem.RemoveHs(m, sanitize=False)
        if m.GetNumAtoms() != n or m.GetNumConformers() == 0:
            continue
        poses.append(m.GetConformer().GetPositions().astype(np.float32))
    return ref, poses


# ----------------------------------------------------------------------------- protein
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "ASH": "D",
       "GLH": "E", "LYN": "K"}


def parse_pdb(path: str):
    """Minimal PDB parser.

    Returns [{'id', 'seq', 'coords': [[n_atoms,3] per residue], 'bfac': [[n_atoms] per residue]}].
    B-factors are the crystallographic displacement parameters: a per-atom experimental measure of how
    mobile that atom is. They are the one dynamics signal present in essentially every structure, and are
    carried here so the model can use observed flexibility rather than a static picture alone.
    """
    chains: dict[str, dict] = {}
    order: list[str] = []
    last_key = None
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break  # first model only
            if not (rec.startswith("ATOM") or (rec.startswith("HETATM") and line[17:20] == "MSE")):
                continue
            resn = line[17:20].strip()
            if resn not in AA3:
                continue
            alt = line[16]
            if alt not in (" ", "A", "1"):
                continue
            elem = line[76:78].strip() or line[12:16].strip()[0]
            if elem.upper() == "H" or line[12:16].strip().startswith("H"):
                continue
            ch = line[21]
            key = (ch, line[22:26], line[26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            try:
                b = float(line[60:66])
            except ValueError:
                b = 0.0
            if ch not in chains:
                chains[ch] = {"id": ch, "seq": [], "coords": [], "bfac": []}
                order.append(ch)
            c = chains[ch]
            if key != last_key:
                c["seq"].append(AA3[resn])
                c["coords"].append([])
                c["bfac"].append([])
                last_key = key
            c["coords"][-1].append(xyz)
            c["bfac"][-1].append(b)
    out = []
    for ch in order:
        c = chains[ch]
        out.append({"id": ch, "seq": "".join(c["seq"]),
                    "coords": [np.asarray(r, dtype=np.float32) for r in c["coords"]],
                    "bfac": [np.asarray(r, dtype=np.float32) for r in c["bfac"]]})
    return out


def extract_pocket(chains, poses, cutoff: float = 10.0, max_res: int = 64):
    """Select residues whose heavy atoms come within `cutoff` Å of any ligand atom (any pose).

    Returns list of (chain_idx, res_idx) and distance tensor D [K, P, N] (min residue-atom distance per pose).
    """
    lig_all = np.concatenate(poses, 0)
    cand = []
    for ci, ch in enumerate(chains):
        for ri, rc in enumerate(ch["coords"]):
            d = np.sqrt(((rc[:, None, :] - lig_all[None, :, :]) ** 2).sum(-1)).min()
            if d < cutoff:
                cand.append((d, ci, ri))
    if not cand:
        return [], None
    cand.sort()
    cand = sorted(cand[:max_res], key=lambda t: (t[1], t[2]))
    sel = [(ci, ri) for _, ci, ri in cand]
    D = np.zeros((len(poses), len(sel), poses[0].shape[0]), dtype=np.float32)
    for k, lig in enumerate(poses):
        for p, (ci, ri) in enumerate(sel):
            rc = chains[ci]["coords"][ri]
            D[k, p] = np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
    return sel, D



def extract_pocket_random(chains, poses, max_res: int = 64, rng=None):
    """CONTROL arm: `max_res` residues drawn uniformly at random from the whole protein, with their REAL
    distances to the ligand.

    `extract_pocket` does two things at once: it picks which residues the model sees, and it hands over the
    geometry of those residues. If structure mode beats the sequence baseline, either could be the reason.
    This keeps the geometry machinery untouched and only destroys the selection, so the two can be told
    apart.
    """
    rng = rng or np.random
    lig_all = np.concatenate(poses, 0)
    allres = [(ci, ri) for ci, ch in enumerate(chains) for ri in range(len(ch["coords"]))]
    if not allres:
        return [], None
    k = min(max_res, len(allres))
    idx = rng.choice(len(allres), size=k, replace=False)
    sel = sorted([allres[i] for i in idx])
    D = np.zeros((len(poses), len(sel), poses[0].shape[0]), dtype=np.float32)
    for k_, lig in enumerate(poses):
        for p_, (ci, ri) in enumerate(sel):
            rc = chains[ci]["coords"][ri]
            D[k_, p_] = np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
    return sel, D


def extract_pocket_wide(chains, poses, cutoff: float = 10.0, max_res: int = 128,
                        n_true: int = 48, rng=None):
    """True contact residues HIDDEN among decoy residues drawn from the rest of the protein.

    `extract_pocket` picks residues by distance, so the question "which residues touch the ligand" is
    answered by the featuriser and never reaches the model. v12 measured the consequence: training with
    every distance set to a constant cost 0.000 RMSE, because the residue list already carried all the
    structural signal the model used.

    Here the model gets `n_true` contact residues plus decoys up to `max_res`, merged into sequence order
    so position gives nothing away. The distance matrix is now the ONLY way to tell contacts from decoys,
    which makes "can this architecture read geometry when it has to" a question the data can answer.
    """
    rng = rng or np.random
    true_sel, _ = extract_pocket(chains, poses, cutoff=cutoff, max_res=n_true)
    if not true_sel:
        return [], None
    allres = [(ci, ri) for ci, ch in enumerate(chains) for ri in range(len(ch["coords"]))]
    pool = [t for t in allres if t not in set(true_sel)]
    n_decoy = min(max(0, max_res - len(true_sel)), len(pool))
    decoys = [pool[i] for i in rng.choice(len(pool), size=n_decoy, replace=False)] if n_decoy else []
    sel = sorted(set(true_sel) | set(decoys))
    D = np.zeros((len(poses), len(sel), poses[0].shape[0]), dtype=np.float32)
    for k, lig in enumerate(poses):
        for p_, (ci, ri) in enumerate(sel):
            rc = chains[ci]["coords"][ri]
            D[k, p_] = np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
    return sel, D

def residue_flexibility(chains, pocket):
    """Per-pocket-residue flexibility features from B-factors, standardised WITHIN the structure.

    Absolute B-factors are not comparable across entries (resolution, refinement protocol and overall
    scaling differ; across BDB2020+ the per-structure mean ranges from 0 to 131). What is comparable is
    how mobile a residue is *relative to the rest of its own structure*, so each feature is a z-score
    against that structure's own atoms.

    Returns [K, 2], or None when the structure's B-factor column is constant (no information).
    """
    allb = np.concatenate([b for c in chains for b in c["bfac"] if len(b)]) if chains else np.zeros(1)
    mu, sd = float(allb.mean()), float(allb.std())
    if sd < 1e-6:
        # Constant B-factor column: the deposited file carries no flexibility information at all (every
        # mpro/*/protein.pdb in this project is like this). Returning zeros would hand the model a constant
        # it never saw in training and silently confound the comparison, so the caller is told instead.
        return None
    out = np.zeros((len(pocket), 2), dtype=np.float32)
    for i, (ci, ri) in enumerate(pocket):
        b = chains[ci]["bfac"][ri]
        if len(b):
            out[i, 0] = (float(b.mean()) - mu) / sd
            out[i, 1] = float(b.std()) / sd
    return np.clip(out, -5.0, 5.0)
