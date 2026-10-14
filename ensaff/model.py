"""EnsAff: AlphaFold-style pair network for protein-ligand binding affinity.

Ideas borrowed from AlphaFold 2 and adapted to a bipartite (protein residue x ligand atom) pair representation:
  * pair representation z[i, j] updated by axial attention along both axes (Evoformer-style)
  * "ligand-triangle" multiplicative update: z[i, j] <- sum_l z[i, l] * e[l, j] using known ligand edges
    (bond type + graph distance), the bipartite analogue of the triangle multiplicative update
  * recycling of z and single representations (stop-gradient on earlier cycles)
  * a per-prediction uncertainty head (Gaussian mean + variance), later calibrated with split-conformal

New element: several protein-ligand states (poses / conformations) are scored separately and combined with a
learnable soft-max over states, pK = tau * logsumexp(pK_k / tau) - tau * log K, which mirrors how a binding free
energy aggregates over an ensemble of bound states (tau is initialised at kT in pK units, ~0.43 at 298 K).

Two protein modes:
  * "seq"    : no structure. A ligand-conditioned latent pocket (Perceiver-style cross-attention of M learned
               slots over all residues) compresses the protein; its attention map is a predicted binding site.
  * "struct" : pocket residues from a structure; residue-atom distances of every pose enter the pair init.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LIG_FEAT_DIM, N_BOND, N_TOPO

NEG = -1e4


def masked_mean(x, mask, dim):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class Transition(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mult * d, d))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- ligand encoder
class LigandEncoder(nn.Module):
    """Edge-aware message passing on the heavy-atom graph."""

    def __init__(self, d, n_layers=3, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(LIG_FEAT_DIM, d)
        self.bond = nn.ModuleList([nn.Embedding(N_BOND, d) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([Transition(d, 2, dropout) for _ in range(n_layers)])

    def forward(self, x, bond, mask):
        h = self.inp(x)
        adj = (bond > 0).unsqueeze(-1).to(h.dtype)                      # [B,N,N,1]
        for emb, norm, mlp in zip(self.bond, self.norm, self.mlp):
            msg = (adj * F.gelu(h.unsqueeze(1) + emb(bond))).sum(2)      # sum_j over neighbours of i
            h = h + mlp(norm(h + msg))
            h = h * mask.unsqueeze(-1)
        return h


# ----------------------------------------------------------------------------- latent pocket (seq mode)
class LatentPocket(nn.Module):
    def __init__(self, d, n_slots, n_heads, n_layers=2, dropout=0.0):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.ctx = nn.Linear(d, d)
        self.lq = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.lkv = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.att = nn.ModuleList([nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
                                  for _ in range(n_layers)])
        self.ff = nn.ModuleList([Transition(d, 2, dropout) for _ in range(n_layers)])

    def forward(self, prot, prot_mask, lig_pool):
        q = self.slots.unsqueeze(0) + self.ctx(lig_pool).unsqueeze(1)
        attn = None
        for lq, lkv, att, ff in zip(self.lq, self.lkv, self.att, self.ff):
            kv = lkv(prot)
            o, attn = att(lq(q), kv, kv, key_padding_mask=~prot_mask, need_weights=True)
            q = q + o
            q = q + ff(q)
        return q, attn                                                   # attn: [B, M, L]


# ----------------------------------------------------------------------------- pair stack
class AxialAttention(nn.Module):
    """Gated self-attention along the last-but-one axis of x [B, R, S, c] (each row attends over S)."""

    def __init__(self, c, n_heads, n_bias=0, dropout=0.0):
        super().__init__()
        self.h, self.dh = n_heads, c // n_heads
        self.ln = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.gate = nn.Linear(c, c)
        self.out = nn.Linear(c, c)
        self.bias = nn.Embedding(n_bias, n_heads) if n_bias else None
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, key_mask, bias_idx=None):
        B, R, S, c = x.shape
        xn = self.ln(x)
        q, k, v = self.qkv(xn).view(B, R, S, 3, self.h, self.dh).unbind(3)
        logits = torch.einsum("brshd,brthd->brhst", q, k).float() / math.sqrt(self.dh)
        if self.bias is not None and bias_idx is not None:
            logits = logits + self.bias(bias_idx).permute(0, 3, 1, 2).unsqueeze(1).float()   # [B,1,H,S,S]
        logits = logits.masked_fill(~key_mask[:, None, None, None, :], NEG)
        a = logits.softmax(-1).to(v.dtype)
        o = torch.einsum("brhst,brthd->brshd", a, v).reshape(B, R, S, c)
        o = torch.sigmoid(self.gate(xn)) * o
        return self.drop(self.out(o))


class LigandTriangleUpdate(nn.Module):
    """z[i,j] += g * W( sum_l a[i,l] * e[l,j] ) with e built from ligand bond type + topological distance."""

    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.a = nn.Linear(c, c)
        self.ag = nn.Linear(c, c)
        self.g = nn.Linear(c, c)
        self.lo = nn.LayerNorm(c)
        self.out = nn.Linear(c, c)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, edge, lig_mask):
        zn = self.ln(z)
        a = self.a(zn) * torch.sigmoid(self.ag(zn)) * lig_mask[:, None, :, None]
        n = lig_mask.sum(1).clamp(min=1).to(z.dtype)[:, None, None, None]
        upd = torch.einsum("bilc,bljc->bijc", a, edge) / n
        return self.drop(torch.sigmoid(self.g(zn)) * self.out(self.lo(upd)))


class PairBlock(nn.Module):
    def __init__(self, d, c, n_heads, dropout=0.0):
        super().__init__()
        self.op_p, self.op_l = nn.Linear(d, c), nn.Linear(d, c)
        self.ln_p, self.ln_l = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att_prot = AxialAttention(c, n_heads, dropout=dropout)
        self.att_lig = AxialAttention(c, n_heads, n_bias=N_TOPO, dropout=dropout)
        self.tri = LigandTriangleUpdate(c, dropout)
        self.trans_z = Transition(c, 2, dropout)
        self.zp, self.zl = nn.Linear(c, d), nn.Linear(c, d)
        self.lnz = nn.LayerNorm(c)
        self.trans_p, self.trans_l = Transition(d, 4, dropout), Transition(d, 4, dropout)

    def forward(self, z, s_p, s_l, pmask, lmask, topo, edge):
        pair_mask = (pmask[:, :, None] & lmask[:, None, :]).unsqueeze(-1).to(z.dtype)
        z = z + self.op_p(self.ln_p(s_p))[:, :, None] + self.op_l(self.ln_l(s_l))[:, None, :]
        z = z + self.att_prot(z.transpose(1, 2), pmask).transpose(1, 2)   # each ligand atom attends over residues
        z = z + self.att_lig(z, lmask, topo)                              # each residue attends over atoms
        z = z + self.tri(z, edge, lmask)
        z = (z + self.trans_z(z)) * pair_mask
        zn = self.lnz(z)
        s_p = s_p + self.zp(masked_mean(zn, lmask[:, None, :].expand(-1, z.shape[1], -1), 2))
        s_l = s_l + self.zl(masked_mean(zn.transpose(1, 2), pmask[:, None, :].expand(-1, z.shape[2], -1), 2))
        s_p = s_p + self.trans_p(s_p)
        s_l = s_l + self.trans_l(s_l)
        return z, s_p, s_l


def rbf(d, n=16, dmax=20.0):
    centers = torch.linspace(0.0, dmax, n, device=d.device, dtype=d.dtype)
    width = dmax / n
    return torch.exp(-((d.unsqueeze(-1) - centers) / width) ** 2)


# ----------------------------------------------------------------------------- full model
class EnsAff(nn.Module):
    def __init__(self, esm_dim, mode="seq", d=128, c=64, n_blocks=4, n_heads=4, n_slots=48, n_rbf=16,
                 n_recycle=3, dropout=0.1, tau_init=0.43):
        super().__init__()
        assert mode in ("seq", "struct")
        self.mode, self.n_recycle, self.n_rbf = mode, n_recycle, n_rbf
        self.prot_in = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, d))
        self.lig_enc = LigandEncoder(d, 3, dropout)
        if mode == "seq":
            self.pocket = LatentPocket(d, n_slots, n_heads, 2, dropout)
        else:
            self.rbf_proj = nn.Linear(n_rbf, c)
        self.zp0, self.zl0 = nn.Linear(d, c), nn.Linear(d, c)
        self.edge_bond, self.edge_topo = nn.Embedding(N_BOND, c), nn.Embedding(N_TOPO, c)
        self.blocks = nn.ModuleList([PairBlock(d, c, n_heads, dropout) for _ in range(n_blocks)])
        self.rec_z, self.rec_p, self.rec_l = (nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c)),
                                              nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)),
                                              nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)))
        for m in (self.rec_z, self.rec_p, self.rec_l):
            nn.init.zeros_(m[1].weight)
            nn.init.zeros_(m[1].bias)
        self.pool = nn.Linear(c, 1)
        self.head = nn.Sequential(nn.LayerNorm(c + 2 * d), nn.Linear(c + 2 * d, d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d, 2))
        self.tau_raw = nn.Parameter(torch.tensor(math.log(math.expm1(tau_init - 0.05))))
        self.out_shift = nn.Parameter(torch.tensor(6.4))  # mean pK of PDBbind; eases early training

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 0.05

    def _trunk(self, z, s_p, s_l, pm, lm, topo, edge):
        for blk in self.blocks:
            z, s_p, s_l = blk(z, s_p, s_l, pm, lm, topo, edge)
        return z, s_p, s_l

    def forward(self, batch, n_cycles=None):
        lm = batch["lig_mask"]
        s_l0 = self.lig_enc(batch["lig_x"], batch["bond"], lm)
        prot = self.prot_in(batch["prot"])
        pocket_attn = None
        if self.mode == "seq":
            s_p0, pocket_attn = self.pocket(prot, batch["prot_mask"], masked_mean(s_l0, lm, 1))
            pm = torch.ones(s_p0.shape[:2], dtype=torch.bool, device=s_p0.device)
        else:
            s_p0, pm = prot, batch["prot_mask"]
        B, M, N = s_p0.shape[0], s_p0.shape[1], s_l0.shape[1]
        z0 = self.zp0(s_p0)[:, :, None] + self.zl0(s_l0)[:, None, :]           # [B,M,N,c]
        if self.mode == "struct":
            dist, smask = batch["dist"], batch["state_mask"]
            K = dist.shape[1]
            z0 = z0[:, None] + self.rbf_proj(rbf(dist, self.n_rbf))             # [B,K,M,N,c]
            z0 = z0.flatten(0, 1)
        else:
            K, smask = 1, torch.ones(B, 1, dtype=torch.bool, device=z0.device)
        rep = lambda t: t.repeat_interleave(K, 0) if K > 1 else t                # noqa: E731
        s_p0, s_l0, pm_k, lm_k = rep(s_p0), rep(s_l0), rep(pm), rep(lm)
        topo, edge = rep(batch["topo"]), rep(self.edge_bond(batch["bond"]) + self.edge_topo(batch["topo"]))

        n = n_cycles or (torch.randint(1, self.n_recycle + 1, (1,)).item() if self.training else self.n_recycle)
        z, s_p, s_l = torch.zeros_like(z0), torch.zeros_like(s_p0), torch.zeros_like(s_l0)
        for cyc in range(n):
            last = cyc == n - 1
            with torch.set_grad_enabled(torch.is_grad_enabled() and last):
                z, s_p, s_l = self._trunk(z0 + self.rec_z(z), s_p0 + self.rec_p(s_p), s_l0 + self.rec_l(s_l),
                                          pm_k, lm_k, topo, edge)
            if not last:
                z, s_p, s_l = z.detach(), s_p.detach(), s_l.detach()

        # per-state affinity
        pair_mask = pm_k[:, :, None] & lm_k[:, None, :]
        w = self.pool(z).squeeze(-1).float().masked_fill(~pair_mask, NEG).flatten(1).softmax(-1)
        zpool = (w.unsqueeze(-1).to(z.dtype) * z.flatten(1, 2)).sum(1)
        h = torch.cat([zpool, masked_mean(s_p, pm_k, 1), masked_mean(s_l, lm_k, 1)], -1)
        o = self.head(h).float()
        mu_k = (o[:, 0] + self.out_shift).view(B, K)
        var_k = (F.softplus(o[:, 1]) + 1e-3).view(B, K)

        # ensemble aggregation (soft-max over states, free-energy style)
        tau = self.tau
        logits = (mu_k / tau).masked_fill(~smask, NEG)
        kcount = smask.sum(1).clamp(min=1).float()
        mu = tau * torch.logsumexp(logits, 1) - tau * torch.log(kcount)
        p = logits.softmax(1)
        mbar = (p * mu_k).sum(1, keepdim=True)
        var = (p * (var_k + (mu_k - mbar) ** 2)).sum(1)
        return {"mu": mu, "var": var, "state_mu": mu_k, "state_w": p, "pocket_attn": pocket_attn}



def pose_hinge(state_mu, smask, n_real, margin: float = 0.5):
    """The crystal pose must outscore its decoys by `margin`.

    Without a term like this nothing in the objective ever makes the per-state predictions differ. v12
    measured the consequence directly: K = 1, 3, 5, 9 states gave identical test RMSE to three decimals,
    and a model trained with every distance set to a constant matched the full model exactly. The state
    axis and the distance channel were both present and both carried nothing.

    This gives the distance channel a job the model cannot do without it. Decoys are generated by
    rotating the ligand about its own centroid, so the coarse distance statistics are unchanged and only
    atom-level geometry separates a decoy from the real pose.
    """
    B, K = state_mu.shape
    idx = torch.arange(K, device=state_mu.device)[None, :]
    real = (idx < n_real[:, None]) & smask
    decoy = (idx >= n_real[:, None]) & smask
    if not bool(decoy.any()):
        return state_mu.sum() * 0.0
    best_real = state_mu.masked_fill(~real, NEG).max(1).values
    viol = (margin - (best_real[:, None] - state_mu)).clamp(min=0) * decoy.float()
    return (viol.sum(1) / decoy.float().sum(1).clamp(min=1)).mean()


def pose_accuracy(state_mu, smask, n_real):
    """Fraction of complexes where every decoy scores below every crystal pose. 0.5-ish means the model
    cannot tell a rotated ligand from the real one; near 1.0 means the distance channel is being read."""
    B, K = state_mu.shape
    idx = torch.arange(K, device=state_mu.device)[None, :]
    real = (idx < n_real[:, None]) & smask
    decoy = (idx >= n_real[:, None]) & smask
    has = decoy.any(1)
    if not bool(has.any()):
        return float("nan")
    worst_real = state_mu.masked_fill(~real, 1e9).min(1).values
    best_decoy = state_mu.masked_fill(~decoy, NEG).max(1).values
    return float(((worst_real > best_decoy) & has).sum().item() / max(1, int(has.sum().item())))

def beta_nll(mu, var, y, beta=0.5):
    """beta-NLL (Seitzer et al., 2022): Gaussian NLL weighted by var^beta to avoid variance collapse."""
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return (nll * var.detach() ** beta).mean()
