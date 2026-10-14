r"""UCUNCU MIMARI: E(n)-degismez graf sinir agi.

Terim hassasiyeti: mesaj gecisi esdegiskendir, cikti bir SKALER oldugu icin
ag butun kompleksin otelenmesi ve dondurulmesi altinda DEGISMEZDIR. Makalede
de bu ayrim boyle yazilmalidir.


NEDEN BU MODUL VAR
------------------
Tartisma bolumunde su hipotezi one surduk: cift-gosterim ve voksel modellerinde
mesafe kanali TOPLAMSAL giriyor ve havuzlamada mesafeye bagli bir esik yok, bu
yuzden hicbir sey bir temasi temas olarak islemeye zorlamiyor; temas-sayim modeli
ise bastan sona mesafe esiginden ibaret ve geometriyi kullanan tek model o.

Bu modul o hipotezin dogrudan testidir. Esdegisken graf aginda geometri
toplamsal degildir: iki dugum arasindaki mesaj, aralarindaki uzakligin bir
FONKSIYONU olarak uretilir (Satorras ve ark.), yani mesaj mesafeyle kapilanir.
Ustelik kenarlar bir yaricap esigiyle kurulur. Boylece temas-sayim modelinin
geometriyi kullanmasini sagladigini dusundugumuz iki ozellik de burada var:
mesafeyle kapilanan mesaj ve mesafeye bagli kesme.

Bu mimari de ayni tanıyı veriyorsa hipotez zayiflar ve bulgu guclenir; vermiyorsa
"geometriyi hangi mimari sinifi kullanir" sorusunun cevabi bulunmus olur. Her iki
sonuc da ise yarar.

DUGUMLER
--------
Cep kalintilari (kalinti basina bir dugum, agir atomlarin agirlik merkezinde) ve
ligand agir atomlari. Dugum ozellikleri yalnizca SINIF kimligidir (bes kalinti
sinifi, bes ligand atom sinifi, bir de protein/ligand bayragi); konum bilgisi
ozelliklerde degil, yalnizca koordinatlardadir. Boylece "geometriyi kaldir"
mudahalesi gercekten yalnizca geometriyi kaldirir.

KOLLAR
------
  eq        gercek koordinatlar
  eq_const  gercek kenarlar, butun mesafeler sabit -> olcu silinir, topoloji kalir
            (cift-gosterim modelindeki const:8'in birebir karsiligi)
  eq_scr    konumlar kutu icinde rastgele -> hem olcu hem topoloji silinir
  eq_rand   rastgele kalinti cebi, gercek koordinatlar

ESDEGISKENLIK
-------------
Model tum kompleksin otelenmesine ve dondurulmesine gore DEGISMEZDIR; bu bir
tasarim ozelligi degil, dogrulanan bir ozelliktir (tools/test_equiv.py). Sondalar
bu yuzden yalnizca BAGIL geometriyi bozanlardir: ligandi kendi icinde oteleme,
kendi agirlik merkezi etrafinda dondurme, olcekleme ve etiket karistirma.

GPU gerekir. Yuksek RAM GEREKMEZ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .contactcount import LIG_CLASS, N_LIG_CLASS, N_RES_CLASS, RES_CLASS, _lig_classes
from .evaluate import regression_metrics
from .features import extract_pocket, extract_pocket_random, featurize_mol, load_ligand_poses, parse_pdb

RADIUS = 10.0          # kenar esigi, angstrom
BOX = 24.0             # scramble kolunda konumlarin cizildigi kutu
N_CLASS = N_RES_CLASS + N_LIG_CLASS


def _rand_rotation(rng):
    q, _ = np.linalg.qr(rng.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def _seeded(cid, spec, salt):
    return np.random.RandomState(
        int(hashlib.md5(f"{salt}{spec}{cid}".encode()).hexdigest()[:8], 16))


def perturb(rec, spec, salt=""):
    """Koordinat uzayinda kontrollu hasar. (xyz, cls, is_lig) dondurur."""
    xyz, cls, lig = rec["xyz"].copy(), rec["cls"].copy(), rec["is_lig"]
    name, _, val = spec.partition(":")
    if name in ("", "none"):
        return xyz, cls, lig
    r = _seeded(rec["id"], spec, salt)
    L = lig.astype(bool)
    if name == "translate":
        xyz[L] += r.randn(1, 3).astype(np.float32) * float(val or 1.0)
    elif name == "rotate":
        c = xyz[L].mean(0, keepdims=True)
        xyz[L] = (xyz[L] - c) @ _rand_rotation(r).T + c
    elif name == "scale":
        c = xyz.mean(0, keepdims=True)
        xyz = ((xyz - c) * float(val) + c).astype(np.float32)
    elif name == "permute_res":
        idx = np.where(~L)[0]
        cls[idx] = cls[idx][r.permutation(len(idx))]
    elif name == "permute_atom":
        idx = np.where(L)[0]
        cls[idx] = cls[idx][r.permutation(len(idx))]
    elif name == "scramble":
        c = rec["centre"].reshape(1, 3)
        xyz = (r.uniform(-BOX / 2, BOX / 2, size=xyz.shape).astype(np.float32) + c)
    else:
        raise ValueError(f"bilinmeyen sonda: {spec}")
    return xyz.astype(np.float32), cls, lig


# --------------------------------------------------------------------------- veri kumesi
class EquivDataset(Dataset):
    """Cep kalintisi agirlik merkezleri + ligand agir atomlari, sinif etiketleriyle."""

    def __init__(self, df, pp_pat, lp_pat, name="", arm="eq", cutoff=10.0, max_res=64,
                 max_atoms=72, cache=None):
        self.arm, self.probe = arm, "none"
        if cache and os.path.exists(cache):
            self.recs = list(np.load(cache, allow_pickle=True)["recs"])
            print(f"[esdegisken:{name}] {len(self.recs)} kompleks (onbellekten), kol={arm}")
            return
        self.recs, skipped = [], 0
        for r in df.itertuples():
            cid = str(r.id)
            pp, lp = pp_pat.format(id=cid), lp_pat.format(id=cid)
            if not (os.path.exists(pp) and os.path.exists(lp)):
                skipped += 1
                continue
            mol, poses = load_ligand_poses(lp, max_poses=1)
            lig = featurize_mol(mol, max_atoms=max_atoms) if mol is not None else None
            if lig is None or not poses:
                skipped += 1
                continue
            chains = parse_pdb(pp)
            if arm == "eq_rand":
                sel, _ = extract_pocket_random(chains, poses, max_res=max_res,
                                               rng=_seeded(cid, "", "pock"))
            else:
                sel, _ = extract_pocket(chains, poses, cutoff=cutoff, max_res=max_res)
            if not sel:
                skipped += 1
                continue
            # kalinti dugumu = agir atomlarinin agirlik merkezi
            pxyz = np.stack([np.asarray(chains[ci]["coords"][ri], np.float32).mean(0)
                             for ci, ri in sel])
            pcls = np.array([RES_CLASS.get(chains[ci]["seq"][ri], 0) for ci, ri in sel],
                            dtype=np.int64)
            lxyz = np.asarray(poses[0], np.float32)[: lig["x"].shape[0]]
            lcls = _lig_cls_safe(np.asarray(lig["x"]))[: len(lxyz)] + N_RES_CLASS
            xyz = np.concatenate([pxyz, lxyz], 0).astype(np.float32)
            cls = np.concatenate([pcls, lcls], 0)
            is_lig = np.concatenate([np.zeros(len(pxyz), np.float32),
                                     np.ones(len(lxyz), np.float32)])
            self.recs.append({"id": cid, "y": float(r.y), "xyz": xyz, "cls": cls,
                              "is_lig": is_lig,
                              "centre": lxyz.mean(0).astype(np.float32)})
        print(f"[esdegisken:{name}] {len(self.recs)} kompleks ({skipped} atlandi), kol={arm}")
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.savez_compressed(cache, recs=np.array(self.recs, dtype=object))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        xyz, cls, lig = r["xyz"], r["cls"], r["is_lig"]
        const = False
        if self.arm == "eq_scr":
            xyz, cls, lig = perturb(r, "scramble", salt="arm")
        elif self.arm == "eq_const":
            const = True          # kenarlar gercek konumlardan, mesafeler sabit
        if self.probe and self.probe != "none":
            rec = {"id": r["id"], "xyz": xyz, "cls": cls, "is_lig": lig,
                   "centre": r["centre"]}
            xyz, cls, lig = perturb(rec, self.probe, salt="probe")
        return {"id": r["id"], "y": np.float32(r["y"]), "xyz": xyz, "cls": cls,
                "is_lig": lig, "const": const}


def _lig_cls_safe(x):
    return _lig_classes(x)


def collate(items):
    B = len(items)
    N = max(len(it["xyz"]) for it in items)
    xyz = torch.zeros(B, N, 3)
    cls = torch.zeros(B, N, dtype=torch.long)
    lig = torch.zeros(B, N)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, it in enumerate(items):
        n = len(it["xyz"])
        xyz[b, :n] = torch.from_numpy(np.asarray(it["xyz"]))
        cls[b, :n] = torch.from_numpy(np.asarray(it["cls"]))
        lig[b, :n] = torch.from_numpy(np.asarray(it["is_lig"]))
        mask[b, :n] = True
    return {"ids": [it["id"] for it in items], "xyz": xyz, "cls": cls, "is_lig": lig,
            "mask": mask, "const": torch.tensor([bool(it["const"]) for it in items]),
            "y": torch.tensor([it["y"] for it in items], dtype=torch.float32)}


# --------------------------------------------------------------------------- model
class EGNNLayer(nn.Module):
    """E(n)-esdegisken mesaj gecisi.

    Geometri YALNIZCA ||x_i - x_j||^2 uzerinden girer ve mesajin ICINDE uretilir,
    yani mesaj mesafeyle kapilanir. Toplamsal bir mesafe kanali yoktur. Kenarlar
    bir yaricap esigiyle sinirlanir, yani mesafeye bagli kesme de vardir.
    """

    def __init__(self, dim, edge_dim=None):
        super().__init__()
        e = edge_dim or dim
        self.phi_e = nn.Sequential(nn.Linear(2 * dim + 1, e), nn.SiLU(), nn.Linear(e, e), nn.SiLU())
        self.phi_h = nn.Sequential(nn.Linear(dim + e, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.gate = nn.Sequential(nn.Linear(e, 1), nn.Sigmoid())

    def forward(self, h, d2, adj):
        B, N, D = h.shape
        hi = h[:, :, None, :].expand(B, N, N, D)
        hj = h[:, None, :, :].expand(B, N, N, D)
        m = self.phi_e(torch.cat([hi, hj, d2[..., None]], -1))
        m = m * self.gate(m) * adj[..., None]
        agg = m.sum(2) / adj.sum(2).clamp(min=1)[..., None]
        return h + self.phi_h(torch.cat([h, agg], -1))


class EquivNet(nn.Module):
    def __init__(self, dim=128, layers=4, dropout=0.15, radius=RADIUS):
        super().__init__()
        self.radius = radius
        self.emb = nn.Embedding(N_CLASS, dim)
        self.flag = nn.Linear(1, dim)
        self.layers = nn.ModuleList([EGNNLayer(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * dim, dim), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(dim, 2))

    def forward(self, b):
        xyz, mask = b["xyz"], b["mask"]
        h = self.emb(b["cls"]) + self.flag(b["is_lig"][..., None])
        d2 = ((xyz[:, :, None, :] - xyz[:, None, :, :]) ** 2).sum(-1)
        pair = mask[:, :, None] & mask[:, None, :]
        adj = ((d2 <= self.radius ** 2) & pair).float()
        adj = adj * (1 - torch.eye(adj.shape[1], device=adj.device))[None]
        if b["const"].any():                 # olcu silinir, topoloji korunur
            c = b["const"].view(-1, 1, 1).float()
            d2 = d2 * (1 - c) + c * (self.radius ** 2) / 4.0
        d2 = d2 / (self.radius ** 2)         # olcek normalizasyonu
        for L in self.layers:
            h = L(h, d2, adj)
        h = self.norm(h) * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        pooled = torch.cat([h.sum(1) / n, h.masked_fill(~mask[..., None], -1e9).amax(1)], -1)
        o = self.head(pooled)
        return o[:, 0], o[:, 1].clamp(-6.0, 4.0)


def beta_nll(mu, logvar, y, beta=0.5):
    var = logvar.exp()
    return ((0.5 * (logvar + (y - mu) ** 2 / var)) * var.detach() ** beta).mean()


# --------------------------------------------------------------------------- egitim
def run_epoch(model, loader, dev, opt=None):
    train = opt is not None
    model.train(train)
    tot, n, P, Y, I = 0.0, 0, [], [], []
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.set_grad_enabled(train):
            mu, lv = model(b)
            loss = beta_nll(mu, lv, b["y"])
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot += float(loss.detach()) * len(b["y"])
        n += len(b["y"])
        P.append(mu.detach().float().cpu().numpy())
        Y.append(b["y"].detach().float().cpu().numpy())
        I += b["ids"]
    return tot / max(n, 1), np.concatenate(P), np.concatenate(Y), I


def train_one(seed, tr, va, te, out, dev, epochs=30, bs=16, lr=3e-4, dim=128, workers=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EquivNet(dim=dim).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    kw = dict(num_workers=workers, collate_fn=collate, pin_memory=(dev.type == "cuda"))
    ltr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **kw)
    lva = DataLoader(va, batch_size=bs, **kw)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        t0 = time.time()
        trl, *_ = run_epoch(model, ltr, dev, opt)
        _, vp, vy, _ = run_epoch(model, lva, dev)
        vr = float(np.sqrt(((vp - vy) ** 2).mean()))
        sched.step(vr)
        if vr < best - 1e-4:
            best, bad = vr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [tohum {seed}] devir {ep + 1}/{epochs} egitim {trl:.4f} "
              f"dogrulama RMSE {vr:.4f} ({time.time() - t0:.0f} sn)", flush=True)
        if bad >= 6:
            print(f"  [tohum {seed}] erken durdu")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    d = os.path.join(out, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": dim, "seed": seed, "n_param": npar},
               os.path.join(d, "model.pt"))
    res = {"seed": seed, "n_param": npar, "val_rmse": best}
    for nm, ds in (("val", va), ("test", te)):
        if ds is None:
            continue
        _, p, y, ids = run_epoch(model, DataLoader(ds, batch_size=bs, **kw), dev)
        res[nm] = regression_metrics(y, p)
        np.savez(os.path.join(d, f"pred_{nm}.npz"), ids=np.array(ids), y=y, p=p)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    print(f"  [tohum {seed}] test RMSE {res.get('test', {}).get('rmse', float('nan')):.4f} "
          f"({npar / 1e6:.3f} M parametre)")
    return res


DEFAULT_PROBES = ("none", "translate:0.5", "translate:1.0", "translate:2.0", "translate:4.0",
                  "translate:8.0", "rotate", "scale:1.5", "permute_res", "permute_atom",
                  "scramble")


def probe_battery(out, ds, dev, probes=DEFAULT_PROBES, bs=32, workers=2, tag="test"):
    seeds = sorted(d for d in os.listdir(out) if d.startswith("seed"))
    if not seeds:
        raise SystemExit(f"{out} altinda kontrol noktasi yok")
    rows, base = {}, {}
    for sd in seeds:
        ck = torch.load(os.path.join(out, sd, "model.pt"), map_location=dev, weights_only=False)
        model = EquivNet(dim=ck["dim"]).to(dev)
        model.load_state_dict(ck["state"])
        for spec in probes:
            ds.probe = spec
            _, p, y, _ = run_epoch(model, DataLoader(ds, batch_size=bs, num_workers=workers,
                                                     collate_fn=collate), dev)
            m = regression_metrics(y, p)
            if spec == "none":
                base[sd], dmu, rb = p, 0.0, 1.0
            else:
                b = base[sd]
                dmu = float(np.abs(p - b).mean())
                rb = float(np.corrcoef(p, b)[0, 1]) if p.std() > 1e-9 and b.std() > 1e-9 else 0.0
            rows.setdefault(spec, []).append((m["rmse"], m["pearson"], dmu, rb))
            print(f"  [{sd}] {spec:16s} RMSE {m['rmse']:.4f}  d|mu| {dmu:.4f}  r_temel {rb:.4f}",
                  flush=True)
    ds.probe = "none"
    lines = [f"ESDEGISKEN SONDA RAPORU ({tag}, n = {len(ds)}, {len(seeds)} model)", "=" * 62,
             f"{'sonda':18s}{'RMSE':>16s}{'Pearson':>16s}{'d|mu|':>10s}{'r_temel':>10s}"]
    agg = {}
    for spec in probes:
        a = np.array(rows[spec])
        agg[spec] = {"rmse": float(a[:, 0].mean()), "rmse_sd": float(a[:, 0].std()),
                     "pearson": float(a[:, 1].mean()), "dmu": float(a[:, 2].mean()),
                     "rbase": float(a[:, 3].mean())}
        lines.append(f"{spec:18s}{a[:, 0].mean():>9.4f} +/-{a[:, 0].std():<5.3f}"
                     f"{a[:, 1].mean():>9.4f} +/-{a[:, 1].std():<5.3f}"
                     f"{a[:, 2].mean():>10.4f}{a[:, 3].mean():>10.4f}")
    cost = agg["scramble"]["rmse"] - agg["none"]["rmse"]
    rb = agg["scramble"]["rbase"]
    uses = (cost > 0.05) and (rb < 0.99)
    lines += ["", f"geometriyi silmenin bedeli (scramble - none) = {cost:+.4f} pK",
              f"bozulmus tahminlerin korelasyonu                =  {rb:.4f}",
              f"KARAR: {'GEOMETRIYI KULLANIYOR' if uses else 'GEOMETRIYI KULLANMIYOR'}"]
    txt = "\n".join(lines)
    open(os.path.join(out, f"probe_report_{tag}.txt"), "w").write(txt + "\n")
    json.dump(agg, open(os.path.join(out, f"probes_{tag}.json"), "w"), indent=2)
    print("\n" + txt)
    return agg


# --------------------------------------------------------------------------- CLI
def build_sets(a, arm):
    from .data import load_lp_pdbbind
    sp = load_lp_pdbbind(a.splits, clean_level=a.clean_level)
    mk = lambda df, nm: EquivDataset(df, a.protein_pattern, a.ligand_pattern, nm, arm=arm,
                                     cutoff=a.cutoff, max_res=a.max_res,
                                     cache=(os.path.join(a.cache, f"{arm}_{nm}.npz")
                                            if a.cache else None))
    return mk(sp["train"], "train"), mk(sp["val"], "val"), mk(sp["test"], "test")


def main(argv=None):
    p = argparse.ArgumentParser(description="E(n)-esdegisken graf agi (ucuncu mimari)")
    p.add_argument("--arm", default="eq", choices=["eq", "eq_const", "eq_scr", "eq_rand"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default="runs/eq")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--splits", default="LP_PDBBind.csv")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--protein_pattern", default="pdb/{id}/{id}_protein.pdb")
    p.add_argument("--ligand_pattern", default="pdb/{id}/{id}_ligand.sdf")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--max_res", type=int, default=64)
    p.add_argument("--cache", default="cache_eq")
    p.add_argument("--probes", default=",".join(DEFAULT_PROBES))
    p.add_argument("--probe-only", dest="probe_only", default=None)
    a = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esdegisken] cihaz = {dev}, yaricap {RADIUS} A, {N_CLASS} dugum sinifi")
    if dev.type != "cuda":
        print("[esdegisken] !! UYARI: GPU bulunamadi, bu kol CPU'da cok yavastir.")

    if a.probe_only:
        _, _, te = build_sets(a, a.arm)
        probe_battery(a.probe_only, te, dev, probes=tuple(a.probes.split(",")),
                      workers=a.workers, tag="test")
        return

    tr, va, te = build_sets(a, a.arm)
    os.makedirs(a.out, exist_ok=True)
    res = [train_one(int(s), tr, va, te, a.out, dev, a.epochs, a.bs, a.lr, a.dim, a.workers)
           for s in a.seeds.split(",")]
    r = np.array([x["test"]["rmse"] for x in res if "test" in x])
    print(f"\n[esdegisken:{a.arm}] test RMSE tohum basina {r.mean():.4f} +/- "
          f"{r.std(ddof=1) if len(r) > 1 else 0.0:.4f} ({len(r)} tohum)")
    json.dump({"arm": a.arm, "per_seed_rmse": r.tolist(), "mean": float(r.mean()),
               "sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0},
              open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    if a.arm == "eq":
        probe_battery(a.out, te, dev, probes=tuple(a.probes.split(",")), workers=a.workers,
                      tag="test")


if __name__ == "__main__":
    main()
