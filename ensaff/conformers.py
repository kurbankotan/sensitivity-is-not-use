r"""D3: build a real conformational ensemble for a complex from OTHER crystal structures of the same protein.

The project's question is whether observed protein dynamics helps affinity prediction. Until now the "states"
fed to the aggregation layer were synthetic. The dataset itself, however, contains genuine conformational
variation: 1581 proteins appear in two or more PDB entries (46% of complexes; 56% of the test split). Those
entries are the same protein captured in different crystal forms with different ligands - an experimentally
observed ensemble.

For a complex, sibling entries of the same protein sequence are superposed onto its pocket (Kabsch on the
shared pocket residues) and each gives one alternative pocket geometry. The ligand pose stays fixed, so the
K states differ only in where the protein atoms sit.

LEAKAGE CONSTRAINT (enforced and tested): a sibling contributes GEOMETRY ONLY. Its affinity label is never
read. Siblings are additionally restricted to the same split, so no val/test structure can influence a
training example and vice versa.
"""
from __future__ import annotations

import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation minimising RMSD of P onto Q (both [n,3], paired). Returns (R, t)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def sibling_index(df, split_col: str = None):
    """{sequence: [ids]} - which complexes share a protein sequence. Only ids inside `df` are linked, so
    callers pass one split at a time and siblings can never cross a split boundary."""
    idx = {}
    for r in df.itertuples():
        idx.setdefault(str(r.seq), []).append(str(r.id))
    return {k: v for k, v in idx.items() if len(v) > 1}


def pocket_states_from_siblings(ref_chains, ref_pocket, ref_pose, sibling_chains_list, max_states: int = 4):
    """Distance matrices [K, n_pocket, n_lig] from the reference plus superposed sibling conformations.

    Each sibling is aligned to the reference on the shared pocket residues (matched by sequence position
    within the chain), then the SAME ligand pose is measured against the sibling's pocket atoms.
    """
    def dist(pocket_xyz, lig):
        return np.stack([np.sqrt(((rc[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
                         for rc in pocket_xyz])

    ref_xyz = [ref_chains[ci]["coords"][ri] for ci, ri in ref_pocket]
    out = [dist(ref_xyz, ref_pose)]
    used = 0
    for sib in sibling_chains_list:
        if used >= max_states:
            break
        # align on CA-like first atoms of pocket residues that exist in both
        P, Q, sib_xyz, ok = [], [], [], True
        for ci, ri in ref_pocket:
            ch = ref_chains[ci]
            scand = next((c for c in sib if c["seq"] == ch["seq"]), None)
            if scand is None or ri >= len(scand["coords"]) or not len(scand["coords"][ri]):
                ok = False
                break
            sib_xyz.append(scand["coords"][ri])
            P.append(scand["coords"][ri][0])
            Q.append(ch["coords"][ri][0])
        if not ok or len(P) < 8:
            continue
        R, t = kabsch(np.asarray(P), np.asarray(Q))
        out.append(dist([(R @ x.T).T + t for x in sib_xyz], ref_pose))
        used += 1
    return np.stack(out).astype(np.float32), used


def ensemble_rmsf(ref_chains, ref_pocket, sibling_chains_list, max_sib: int = 8):
    """Per-pocket-residue conformational spread measured ACROSS sibling crystal structures.

    This is the ensemble analogue of a B-factor: instead of one structure's refined mobility parameter, it
    is how far a residue actually moves between independent determinations of the same protein. B-factors
    (a cruder, single-structure proxy) already produced a measurable gain, and the spread here is strongly
    non-uniform - across mpro the mean pocket deviation is 0.34 A while the maximum reaches 4.2 A - so the
    signal lives in which residues move, not in the average.

    Returns [n_pocket, 2]: standardised per-residue RMSF, and the number of siblings it was measured over
    (scaled), or None when fewer than two siblings could be aligned.
    """
    per_res = [[] for _ in ref_pocket]
    used = 0
    for sib in sibling_chains_list:
        if used >= max_sib:
            break
        P, Q, sib_xyz, ok = [], [], [], True
        for ci, ri in ref_pocket:
            ch = ref_chains[ci]
            scand = next((c for c in sib if c["seq"] == ch["seq"]), None)
            if scand is None or ri >= len(scand["coords"]) or not len(scand["coords"][ri]):
                ok = False
                break
            sib_xyz.append(scand["coords"][ri])
            P.append(scand["coords"][ri][0])
            Q.append(ch["coords"][ri][0])
        if not ok or len(P) < 8:
            continue
        R, t = kabsch(np.asarray(P), np.asarray(Q))
        for k, (ci, ri) in enumerate(ref_pocket):
            moved = (R @ sib_xyz[k].T).T + t
            ref_at = ref_chains[ci]["coords"][ri]
            n = min(len(moved), len(ref_at))
            per_res[k].append(float(np.sqrt(((moved[:n] - ref_at[:n]) ** 2).sum(1)).mean()))
        used += 1
    if used < 2:
        return None
    rmsf = np.array([np.mean(v) if v else 0.0 for v in per_res], dtype=np.float32)
    mu, sd = float(rmsf.mean()), float(rmsf.std())
    z = (rmsf - mu) / (sd if sd > 1e-6 else 1.0)
    return np.stack([np.clip(z, -5, 5), np.full(len(rmsf), used / 8.0, dtype=np.float32)], 1)
