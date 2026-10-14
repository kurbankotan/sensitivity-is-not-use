"""Datasets: sequence mode (sequence + SMILES) and structure mode (pocket + ligand poses)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (MAX_TOPO, extract_pocket, extract_pocket_random, extract_pocket_wide, featurize_mol, featurize_smiles, load_ligand_poses,
                       parse_pdb, residue_flexibility)

LP_URL = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv"
BDB_URL = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/BDB2020+.tgz"
MPRO_URL = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/mpro.tgz"
EGFR_URL = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/EGFR.tgz"


# ----------------------------------------------------------------------------- tables
def load_lp_pdbbind(path: str = "LP_PDBBind.csv", clean_level: str = "CL1", drop_covalent: bool = True):
    """LP-PDBBind leak-proof splits (Li et al.). Returns dict split -> DataFrame[id, smiles, seq, y]."""
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(LP_URL, path)
    d = pd.read_csv(path).rename(columns={"Unnamed: 0": "id", "value": "y"})
    if clean_level:
        d = d[d[clean_level].astype(bool)]
    if drop_covalent:
        d = d[~d["covalent"].astype(bool)]
    d = d[~((d["new_split"] == "val") & d["remove_for_balancing_val"].astype(bool))]
    d = d.dropna(subset=["smiles", "seq", "y"])
    return {s: d[d["new_split"] == s][["id", "smiles", "seq", "y"]].reset_index(drop=True)
            for s in ["train", "val", "test"]}


def load_bdb2020(root: str = "BDB2020+"):
    """BDB2020+ external test set (structures released after PDBbind 2020)."""
    if not os.path.exists(root):
        import tarfile
        import urllib.request
        urllib.request.urlretrieve(BDB_URL, "BDB2020+.tgz")
        tarfile.open("BDB2020+.tgz").extractall(".")
    d = pd.read_csv(os.path.join(root, "BDB2020+.csv")).rename(columns={"pdbid": "id", "pKa": "y"})
    return d[["id", "smiles", "seq", "y"]].reset_index(drop=True)


def split_chains(seq: str) -> list[str]:
    return [c for c in str(seq).split(":") if c]


# ----------------------------------------------------------------------------- sequence mode
class SeqAffinityDataset(Dataset):
    """Protein = full chain sequence(s) -> per-residue embeddings; ligand = 2D graph from SMILES."""

    def __init__(self, df: pd.DataFrame, embedder, max_atoms: int = 72, name: str = ""):
        self.recs = []
        chains_all = []
        skipped = 0
        for r in df.itertuples():
            lig = featurize_smiles(r.smiles, max_atoms=max_atoms)
            chains = split_chains(r.seq)
            if lig is None or not chains:
                skipped += 1
                continue
            self.recs.append({"id": r.id, "y": float(r.y), "chains": chains, "lig": lig})
            chains_all += chains
        embedder.embed_chains(chains_all)
        self.emb = embedder
        print(f"[data:{name}] {len(self.recs)} complexes ({skipped} skipped)")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        prot = np.concatenate([self.emb.get(c) for c in r["chains"]], 0).astype(np.float32)
        return {"id": r["id"], "y": r["y"], "prot": prot, **r["lig"], "dist": None}



# ----------------------------------------------------------------------------- distance-channel probes
def _rand_rotation(rng):
    """Uniform random rotation (QR of a Gaussian matrix, reflection removed)."""
    q, r = np.linalg.qr(rng.randn(3, 3))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def _recompute_D(pocket_xyz, lig):
    return np.stack([np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
                     for rc in pocket_xyz]).astype(np.float32)[None]


def _perturb_D(D, r, spec: str, salt: str = "", rng=None):
    """Controlled damage to the residue-atom distance channel. `spec` is name or name:value.

      none            unchanged
      const:v         every distance set to v        -> geometry removed (RBF term becomes a constant bias)
      permute_res     residue axis shuffled          -> right distances, wrong residues
      permute_atom    ligand-atom axis shuffled      -> right distances, wrong atoms
      scale:f         all distances multiplied by f
      offset:v        v angstrom added to every distance
      translate:s     ligand rigidly displaced, s = per-axis sd (needs coordinates)
      rotate          ligand rotated about its own centroid (needs coordinates)

    Every variant is seeded from the complex id, so the same spec gives the same perturbation on every run
    and two models can be compared on identical damage.
    """
    name, _, val = spec.partition(":")
    if rng is None:   # stable per-complex seed, so a probe repeats exactly across runs and models
        rng = np.random.RandomState(int(hashlib.md5(f"{salt}{name}{val}{r['id']}".encode()).hexdigest()[:8], 16))
    if name == "none":
        return D
    if name == "const":
        return np.full_like(D, float(val or 8.0))
    if name == "scale":
        return (D * float(val)).astype(np.float32)
    if name == "offset":
        return (D + float(val)).astype(np.float32)
    if name == "permute_res":
        return D[:, rng.permutation(D.shape[1]), :]
    if name == "permute_atom":
        return D[:, :, rng.permutation(D.shape[2])]
    if name in ("translate", "rotate"):
        if r.get("pose0") is None or r.get("pocket_xyz") is None:
            raise ValueError(f"'{name}' sondasi koordinat ister; veri kumesini keep_coords=True ile kurun")
        lig = r["pose0"]
        if name == "translate":
            lig = lig + rng.randn(1, 3).astype(np.float32) * float(val)
        else:
            c = lig.mean(0, keepdims=True)
            lig = (lig - c) @ _rand_rotation(rng).T + c
        return _recompute_D(r["pocket_xyz"], lig)
    raise ValueError(f"bilinmeyen sonda: {spec}")


# ----------------------------------------------------------------------------- structure mode
class StructAffinityDataset(Dataset):
    """Protein = binding-pocket residues (ESM embeddings of their chain); ligand = graph from SDF;
    states = ligand poses (all poses in the SDF, optionally plus jittered copies)."""

    def __init__(self, df: pd.DataFrame, embedder, protein_pattern: str, ligand_pattern: str,
                 cutoff: float = 10.0, max_res: int = 64, max_atoms: int = 72, max_poses: int = 4,
                 jitter_states: int = 0, jitter_sigma: float = 0.5, name: str = "", eval_states: int = 0,
                 use_bfactor: bool = False, geom: str = "real", pocket_mode: str = "nearest",
                 keep_coords: bool = False, decoy_spec: str = "translate", n_true: int = 48):
        self.recs, chains_all, skipped = [], [], 0
        degenerate = 0
        self.jitter_states, self.jitter_sigma = jitter_states, jitter_sigma
        # eval_states > 0 turns the multi-state aggregation on at INFERENCE too. With a single crystal pose
        # the aggregation is mathematically the identity (K=1), so without this it is only ever exercised as
        # training-time augmentation and its effect on predictions cannot be measured.
        self.eval_states = eval_states
        self.use_bfactor = use_bfactor
        # geom: training-time control over the distance channel; probe: inference-time perturbation.
        # Both go through _perturb_D so a control arm and a probe cannot disagree about what "const" means.
        self.geom, self.pocket_mode, self.probe = geom, pocket_mode, "none"
        # decoy_spec shapes the extra TRAINING states. "rotate" turns the ligand about its own centroid,
        # which leaves every residue-to-centroid distance untouched and changes only which atom faces
        # which residue, so a model that can tell a decoy apart must be reading atom-level geometry
        # rather than a shift in the overall distance distribution.
        self.decoy_spec = decoy_spec
        for r in df.itertuples():
            pp, lp = protein_pattern.format(id=r.id), ligand_pattern.format(id=r.id)
            if not (os.path.exists(pp) and os.path.exists(lp)):
                skipped += 1
                continue
            mol, poses = load_ligand_poses(lp, max_poses=max_poses)
            lig = featurize_mol(mol, max_atoms=max_atoms) if mol is not None else None
            if lig is None or not poses:
                skipped += 1
                continue
            chains = parse_pdb(pp)
            if pocket_mode in ("random", "wide"):
                prng = np.random.RandomState(int(hashlib.md5(f"pock{r.id}".encode()).hexdigest()[:8], 16))
                if pocket_mode == "random":
                    sel, D = extract_pocket_random(chains, poses, max_res=max_res, rng=prng)
                else:
                    sel, D = extract_pocket_wide(chains, poses, cutoff=cutoff, max_res=max_res,
                                                 n_true=n_true, rng=prng)
            else:
                sel, D = extract_pocket(chains, poses, cutoff=cutoff, max_res=max_res)
            if not sel:
                skipped += 1
                continue
            seqs = [c["seq"] for c in chains]
            flex = residue_flexibility(chains, sel) if use_bfactor else None
            if use_bfactor and flex is None:
                degenerate += 1
                flex = np.zeros((len(sel), 2), dtype=np.float32)
            keep_xyz = (jitter_states > 0 or eval_states > 0 or keep_coords
                        or geom in ("translate", "rotate"))
            self.recs.append({"id": r.id, "y": float(r.y), "chain_seqs": seqs, "pocket": sel, "D": D,
                              "lig": lig, "flex": flex,
                              "pose0": poses[0] if keep_xyz else None,
                              "pocket_xyz": [chains[ci]["coords"][ri] for ci, ri in sel] if keep_xyz else None})
            chains_all += seqs
        embedder.embed_chains(chains_all)
        self.emb = embedder
        self.train_mode = False
        msg = f"[data:{name}] {len(self.recs)} complexes ({skipped} skipped)"
        if geom != "real" or pocket_mode != "nearest":
            msg += f"  [KONTROL KOLU geom={geom} pocket={pocket_mode}]"
        if use_bfactor and degenerate:
            msg += (f"  !! UYARI: {degenerate}/{len(self.recs)} yapida B-faktor sutunu SABIT "
                    f"(bilgi yok, sifirlandi) -> bu sette B-faktor karsilastirmasi gecersiz")
        print(msg)
        self.n_degenerate_bfactor = degenerate

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        prot = np.stack([self.emb.get(r["chain_seqs"][ci])[ri] for ci, ri in r["pocket"]]).astype(np.float32)
        if r.get("flex") is not None:          # observed flexibility appended to the residue embedding
            prot = np.concatenate([prot, r["flex"]], axis=1)
        D = r["D"]
        n_real = D.shape[0]          # states 0..n_real-1 are crystal poses; the rest are decoys
        n_extra = self.jitter_states if self.train_mode else self.eval_states
        if n_extra > 0 and r.get("pose0") is not None and self.decoy_spec != "translate":
            # fresh decoys every epoch in training, repeatable ones at evaluation
            rng = np.random if self.train_mode else None
            extra = [_perturb_D(D[:1], r, self.decoy_spec, salt=f"decoy{_i}", rng=rng)[0]
                     for _i in range(n_extra)]
            D = np.concatenate([D, np.stack(extra)], 0)
        elif n_extra > 0 and r.get("pose0") is not None:
            # Perturb the LIGAND COORDINATES and recompute distances. Adding i.i.d. noise straight to the
            # distance matrix (what this did before) is not realisable by any rigid or flexible pose change,
            # so it could not stand in for an alternative bound state. A rigid-body jiggle of the ligand can.
            if self.train_mode:
                rng = np.random
            else:   # stable per-complex seed: Python's hash() is randomised per process
                rng = np.random.RandomState(int(hashlib.md5(str(r["id"]).encode()).hexdigest()[:8], 16))
            extra = []
            for _ in range(n_extra):
                shift = rng.randn(1, 3).astype(np.float32) * self.jitter_sigma
                lig = r["pose0"] + shift                      # rigid translation of the whole ligand
                dd = np.stack([np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
                               for rc in r["pocket_xyz"]])
                extra.append(dd.astype(np.float32))
            D = np.concatenate([D, np.stack(extra)], 0)
        if self.geom != "real":
            D = _perturb_D(D, r, self.geom, salt="geom")
        if self.probe and self.probe != "none":
            D = _perturb_D(D, r, self.probe, salt="probe")
        return {"id": r["id"], "y": r["y"], "prot": prot, **r["lig"], "dist": D, "n_real": n_real}


# ----------------------------------------------------------------------------- collate
def collate(items):
    B = len(items)
    L = max(it["prot"].shape[0] for it in items)
    N = max(it["x"].shape[0] for it in items)
    E = items[0]["prot"].shape[1]
    F = items[0]["x"].shape[1]
    prot = torch.zeros(B, L, E)
    pmask = torch.zeros(B, L, dtype=torch.bool)
    x = torch.zeros(B, N, F)
    lmask = torch.zeros(B, N, dtype=torch.bool)
    bond = torch.zeros(B, N, N, dtype=torch.long)
    topo = torch.full((B, N, N), MAX_TOPO + 1, dtype=torch.long)
    y = torch.tensor([it["y"] for it in items], dtype=torch.float32)
    has_dist = items[0]["dist"] is not None
    if has_dist:
        K = max(it["dist"].shape[0] for it in items)
        dist = torch.zeros(B, K, L, N)
        smask = torch.zeros(B, K, dtype=torch.bool)
    for b, it in enumerate(items):
        l, n = it["prot"].shape[0], it["x"].shape[0]
        prot[b, :l] = torch.from_numpy(it["prot"])
        pmask[b, :l] = True
        x[b, :n] = torch.from_numpy(it["x"])
        lmask[b, :n] = True
        bond[b, :n, :n] = torch.from_numpy(it["bond"].astype(np.int64))
        topo[b, :n, :n] = torch.from_numpy(it["topo"].astype(np.int64))
        if has_dist:
            k = it["dist"].shape[0]
            dist[b, :k, :l, :n] = torch.from_numpy(it["dist"])
            smask[b, :k] = True
    batch = {"ids": [it["id"] for it in items], "prot": prot, "prot_mask": pmask, "lig_x": x, "lig_mask": lmask,
             "bond": bond, "topo": topo, "y": y}
    if has_dist:
        batch["dist"], batch["state_mask"] = dist, smask
        batch["n_real"] = torch.tensor([it.get("n_real", 1) for it in items], dtype=torch.long)
    return batch


# ----------------------------------------------------------------------------- extra external test sets
def _ensure(url: str, tgz: str, root: str):
    if not os.path.exists(root):
        import tarfile
        import urllib.request
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(url, tgz)
        tarfile.open(tgz).extractall(".")
    return os.path.exists(root)


def _seq_from_pdb(path: str) -> str:
    """Chain sequences joined with ':' - the format split_chains() expects."""
    return ":".join(c["seq"] for c in parse_pdb(path) if len(c["seq"]) >= 20)


def load_mpro(root: str = "mpro"):
    """SARS-CoV-2 main protease: 40 ligands against ONE target. Tests ligand ranking within a target,
    which is the lead-optimisation question, as opposed to BDB2020+ which tests new targets."""
    if not _ensure(MPRO_URL, "mpro.tgz", root):
        return None, None, None
    d = pd.read_csv(os.path.join(root, "info.csv"))
    col = "pKa_reliable" if "pKa_reliable" in d.columns else "pKa_full"
    rows = []
    for r in d.itertuples():
        pp = os.path.join(root, str(r.pdbid), "protein.pdb")
        lp = os.path.join(root, str(r.pdbid), "ligand.sdf")
        y = getattr(r, col, None)
        if not (os.path.exists(pp) and os.path.exists(lp)) or pd.isna(y):
            continue
        rows.append({"id": str(r.pdbid), "smiles": r.smiles, "seq": _seq_from_pdb(pp), "y": float(y)})
    df = pd.DataFrame(rows)
    return df, os.path.join(root, "{id}", "protein.pdb"), os.path.join(root, "{id}", "ligand.sdf")


def load_egfr(root: str = "EGFR"):
    """EGFR kinase: 23 ligands against one target (several point mutants)."""
    if not _ensure(EGFR_URL, "EGFR.tgz", root):
        return None, None, None
    d = pd.read_csv(os.path.join(root, "info.csv"))
    rows = []
    for r in d.itertuples():
        pid = str(r.pdbids)
        pp = os.path.join(root, "protein", f"{pid}.pdb")
        lp = os.path.join(root, "ligand", f"{pid}.sdf")
        if not (os.path.exists(pp) and os.path.exists(lp)) or pd.isna(r.value):
            continue
        rows.append({"id": pid, "smiles": r.smiles, "seq": _seq_from_pdb(pp), "y": float(r.value)})
    df = pd.DataFrame(rows)
    return df, os.path.join(root, "protein", "{id}.pdb"), os.path.join(root, "ligand", "{id}.sdf")


def load_external_sets(data_dir: str = ".", exclude_ids=None, clean_level: str = "CL1"):
    """External test sets, as {name: (df, protein_pattern, ligand_pattern)}.

    Complexes that also appear in the LP-PDBBind val/test splits are REMOVED. Without this the EGFR set
    is not external at all: 11 of its 23 entries are the test split re-scored and 2 are in val, which is
    both the early-stopping and the conformal-calibration set. Keeping them would double-count evidence
    and, for the val ones, score the model on data used to select it.
    """
    if exclude_ids is None:
        try:
            tabs = load_lp_pdbbind(os.path.join(data_dir, "LP_PDBBind.csv"), clean_level=clean_level)
            exclude_ids = set(pd.concat([tabs["val"], tabs["test"]])["id"].astype(str).str.lower())
        except Exception:
            exclude_ids = set()
    out = {}
    bdb = load_bdb2020(os.path.join(data_dir, "BDB2020+"))
    bdb_root = os.path.join(data_dir, "BDB2020+", "dataset")
    cand = [("BDB2020+", bdb, os.path.join(bdb_root, "{id}", "protein.pdb"),
             os.path.join(bdb_root, "{id}", "ligand.sdf"))]
    for name, fn, sub in (("mpro", load_mpro, "mpro"), ("EGFR", load_egfr, "EGFR")):
        try:
            df, pp, lp = fn(os.path.join(data_dir, sub))
            if df is not None and len(df):
                cand.append((name, df, pp, lp))
        except Exception as e:
            print(f"[data] {name} yuklenemedi: {e}")
    for name, df, pp, lp in cand:
        n0 = len(df)
        keep = ~df["id"].astype(str).str.lower().isin(exclude_ids)
        df = df[keep].reset_index(drop=True)
        if n0 != len(df):
            print(f"[data] {name}: LP val/test ile ortusen {n0 - len(df)} kompleks cikarildi "
                  f"-> {len(df)} kaldi (dis set olarak gecerli)")
        if len(df) >= 10:
            out[name] = (df, pp, lp)
        else:
            print(f"[data] {name}: tekillestirme sonrasi yalnizca {len(df)} kompleks kaldi -> KULLANILMIYOR")
    return out
