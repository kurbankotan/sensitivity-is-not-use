r"""IKINCI MIMARI: voksel tabanli 3B evrisimli ag.


NEDEN BU MODUL VAR
------------------
v12, cift-gosterim modelinde uc boyutlu geometrinin olculebilir hicbir katki
saglamadigini gosterdi. Buna verilebilecek en guclu hakem itirazi sudur:
"katki sifir cikti cunku mimarin geometriyi okumak icin kurulmamis; daha iyi
bir uzamsal onyargiya sahip bir model farkli davranirdi."

Bu modul o itirazi kapatmak icin yazildi. Burada egitilen model, cift-gosterim
modeliyle SADECE girdi kompleksini paylasir:

  * girdi bir mesafe matrisi degil, bir uzamsal isgal izgarasi (3B voksel),
  * dil modeli gomulmesi yok; kanallar atom/kalinti sinifi,
  * dikkat yok; yalnizca 3B evrisim, yani tum tasarim varsayimi uzamsaldir,
  * ayri egitim dongusu, ayri parametreler.

Eger uzamsal evrisimin tum amaci konumu okumak olan bir mimari de ayni taniyi
veriyorsa, bulgu "bu model" olmaktan cikip "bu model sinifi" olur.

SONDA BATARYASI KOORDINAT UZAYINDA
----------------------------------
Sondalar mesafe matrisine degil koordinatlara uygulanir. Adlar, anlami ayni
kalan yerlerde ayni tutuldu; farkli oldugu tek yerde yeni bir ad verildi:

  translate:s   ligandin katı yer degistirmesi            (ayni anlam)
  rotate        ligandin kendi agirlik merkezi etrafinda  (ayni anlam)
  scale:s       tum koordinatlarin merkeze gore olceklenmesi
  permute_res   kalinti SINIF etiketlerinin karistirilmasi, konumlar yerinde
                (cift-gosterim modelindeki "mesafeler yanlis kalintiya bagli"
                 hasarinin birebir karsiligi)
  permute_atom  ligand atom sinif etiketlerinin karistirilmasi
  scramble      her atomun kutu icinde rastgele bir konuma tasinmasi, bilesim
                korunur. Bu, "const:8"in voksel karsiligidir: ne varsa durur,
                nerede oldugu silinir. Mesafe kavrami olmadigi icin ad da
                farklidir.

Hasar, her komplekste karmadan tureyen sabit bir tohumla uretilir, boylece bir
sonda calistirmadan calistirmaya ve modelden modele birebir tekrarlanir.

KOLLAR
------
  vx        gercek koordinatlar
  vx_scr    scramble ile SIFIRDAN egitim (geometrinin ogrenilebilir katkisi)
  vx_rand   rastgele kalinti cebi, gercek koordinatlar (cep secimi)

KULLANIM
--------
  python -m ensaff.voxel --arm vx --seeds 0,1,2,3,4 --out runs/vx
  python -m ensaff.voxel --probe-only runs/vx --probes none,translate:2.0,scramble

GPU gerekir (CPU'da da calisir ama cok yavastir). Yuksek RAM GEREKMEZ:
koordinatlar onbellege alinir, izgara her ornekte anlik kurulur.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .contactcount import LIG_CLASS, N_LIG_CLASS, N_RES_CLASS, RES_CLASS
from .evaluate import regression_metrics
from .features import ELEMENTS, extract_pocket, extract_pocket_random, featurize_mol, load_ligand_poses, parse_pdb

GRID = 24          # kenar uzunlugu, voksel sayisi
SPACING = 1.0      # angstrom / voksel
N_CH = N_RES_CLASS + N_LIG_CLASS


# --------------------------------------------------------------------------- koordinat uzayinda hasar
def _rand_rotation(rng):
    q, _ = np.linalg.qr(rng.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def _seeded(cid: str, spec: str, salt: str):
    h = hashlib.md5(f"{salt}{spec}{cid}".encode()).hexdigest()[:8]
    return np.random.RandomState(int(h, 16))


def perturb_coords(rec: dict, spec: str, salt: str = "", rng=None):
    """Kontrollu hasar. `rec`i degistirmez; (prot_xyz, prot_cls, lig_xyz, lig_cls) dondurur."""
    px, pc = rec["prot_xyz"], rec["prot_cls"]
    lx, lc = rec["lig_xyz"], rec["lig_cls"]
    name, _, val = spec.partition(":")
    if name in ("", "none"):
        return px, pc, lx, lc
    r = rng if rng is not None else _seeded(rec["id"], spec, salt)
    if name == "translate":
        return px, pc, lx + r.randn(1, 3).astype(np.float32) * float(val or 1.0), lc
    if name == "rotate":
        c = lx.mean(0, keepdims=True)
        return px, pc, ((lx - c) @ _rand_rotation(r).T + c).astype(np.float32), lc
    if name == "scale":
        s = float(val)
        c = np.concatenate([px, lx], 0).mean(0, keepdims=True)
        return ((px - c) * s + c).astype(np.float32), pc, ((lx - c) * s + c).astype(np.float32), lc
    if name == "permute_res":
        # konumlar yerinde, kalinti sinifi etiketleri kaydirilir
        return px, pc[r.permutation(len(pc))], lx, lc
    if name == "permute_atom":
        return px, pc, lx, lc[r.permutation(len(lc))]
    if name == "scramble":
        # "const:8"in karsiligi: bilesim durur, konum silinir.
        # Ornekleme araligi kutudan bir voksel ICERIDE tutulur. Tam kenarda ornekleyen
        # bir atomun trilineer agirliginin yarisi izgaranin disina duser; bu, karistirilmis
        # kolun toplam kutlesini gercek kola gore dusururdu ve geometriyle ilgisi olmayan
        # bir bilesim farki yaratirdi. Icerideki aralik iki kolu ayni toplam kutlede tutar.
        c = np.asarray(rec["centre"], dtype=np.float32).reshape(1, 3)
        lim = GRID * SPACING / 2.0 - SPACING
        npx = (r.uniform(-lim, lim, size=px.shape).astype(np.float32) + c)
        nlx = (r.uniform(-lim, lim, size=lx.shape).astype(np.float32) + c)
        return npx, pc, nlx, lc
    raise ValueError(f"bilinmeyen sonda: {spec}")


# --------------------------------------------------------------------------- izgara
def splat(prot_xyz, prot_cls, lig_xyz, lig_cls, centre) -> np.ndarray:
    """Trilineer dagitim. SABIT bir merkeze oturtulmus [N_CH, G, G, G] izgara.

    `centre` bozulmamis ligandin agirlik merkezidir ve kayit kurulurken bir kez
    hesaplanir. Izgarayi o ana ligandin merkezine oturtmak cazip gorunur ama
    yanlistir: o zaman translate sondasi cerceveyi de birlikte tasir, sonda
    kismen kendini goturur ve kutunun disina tasan atomlar yuzunden "geometri
    degisti" ile "bilesim degisti" birbirine karisir. Sabit cerceve ikisini
    ayirir.

    Sert (ikili) isgal yerine trilineer dagitim kullanilir: bir voksel genisliginden
    kucuk yer degistirmeler boylece girdide GORUNUR kalir. Modele geometriyi okumak
    icin mumkun olan en iyi sans verilmis olur, ki bulgu "izgara cok kabaydi"
    diye aciklanamasin.
    """
    g = np.zeros((N_CH, GRID, GRID, GRID), dtype=np.float32)
    centre = np.asarray(centre, dtype=np.float32).reshape(3)
    half = GRID / 2.0
    for xyz, cls, off in ((prot_xyz, prot_cls, 0), (lig_xyz, lig_cls, N_RES_CLASS)):
        if len(xyz) == 0:
            continue
        p = (xyz - centre) / SPACING + half - 0.5          # voksel koordinatlari
        i0 = np.floor(p).astype(np.int64)
        frac = (p - i0).astype(np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.where(dx, frac[:, 0], 1 - frac[:, 0]) *
                         np.where(dy, frac[:, 1], 1 - frac[:, 1]) *
                         np.where(dz, frac[:, 2], 1 - frac[:, 2]))
                    idx = i0 + np.array([dx, dy, dz])
                    ok = ((idx >= 0).all(1) & (idx < GRID).all(1) & (w > 0))
                    if not ok.any():
                        continue
                    np.add.at(g, (cls[ok] + off, idx[ok, 0], idx[ok, 1], idx[ok, 2]), w[ok])
    return g


# --------------------------------------------------------------------------- veri kumesi
def _lig_cls_from_x(x: np.ndarray) -> np.ndarray:
    idx = x[:, :len(ELEMENTS) + 1].argmax(1)
    out = np.full(len(idx), N_LIG_CLASS - 1, dtype=np.int64)
    for i, e in enumerate(ELEMENTS):
        out[idx == i] = LIG_CLASS.get(e, N_LIG_CLASS - 1)
    return out


class VoxelDataset(Dataset):
    """Cep agir atom koordinatlari + ligand koordinatlari; izgara her ornekte kurulur.

    Onbellekte tutulan sey koordinatlardir (kompleks basina birkac KB), izgara degil.
    Bu yuzden YUKSEK RAM GEREKMEZ ve sondalar koordinat uzayinda uygulanabilir.
    """

    def __init__(self, df, protein_pattern: str, ligand_pattern: str, cutoff: float = 10.0,
                 max_res: int = 64, max_atoms: int = 72, name: str = "", arm: str = "vx",
                 cache: str | None = None):
        self.arm, self.probe = arm, "none"
        if cache and os.path.exists(cache):
            self.recs = list(np.load(cache, allow_pickle=True)["recs"])
            print(f"[voksel:{name}] {len(self.recs)} kompleks (onbellekten)")
            return
        self.recs, skipped, clipped = [], 0, 0
        for r in df.itertuples():
            pp, lp = protein_pattern.format(id=r.id), ligand_pattern.format(id=r.id)
            if not (os.path.exists(pp) and os.path.exists(lp)):
                skipped += 1
                continue
            mol, poses = load_ligand_poses(lp, max_poses=1)
            lig = featurize_mol(mol, max_atoms=max_atoms) if mol is not None else None
            if lig is None or not poses:
                skipped += 1
                continue
            chains = parse_pdb(pp)
            if arm == "vx_rand":
                prng = _seeded(str(r.id), "", "pock")
                sel, _ = extract_pocket_random(chains, poses, max_res=max_res, rng=prng)
            else:
                sel, _ = extract_pocket(chains, poses, cutoff=cutoff, max_res=max_res)
            if not sel:
                skipped += 1
                continue
            xyz, cls = [], []
            for ci, ri in sel:
                rc = chains[ci]["coords"][ri]
                k = RES_CLASS.get(chains[ci]["seq"][ri], 0)
                xyz.append(np.asarray(rc, dtype=np.float32))
                cls.append(np.full(len(rc), k, dtype=np.int64))
            lx = np.asarray(poses[0], dtype=np.float32)[: lig["x"].shape[0]]
            # Kutuya sigmayan protein atomlari burada atilir. Boylece gercek kol ile
            # karistirilmis kol tam olarak ayni atom cokluguyla calisir ve aralarindaki
            # tek fark konum olur.
            px_all, pc_all = np.concatenate(xyz, 0), np.concatenate(cls, 0)
            centre = lx.mean(0).astype(np.float32)
            keep = (np.abs(px_all - centre) <= GRID * SPACING / 2.0 - SPACING).all(1)
            px_all, pc_all = px_all[keep], pc_all[keep]
            if len(px_all) == 0:
                skipped += 1
                continue
            if not (np.abs(lx - centre) <= GRID * SPACING / 2.0 - SPACING).all():
                clipped += 1        # ligand kutuya sigmiyor; sayilir ve bildirilir
            self.recs.append({"id": str(r.id), "y": float(r.y), "centre": centre,
                              "prot_xyz": px_all,
                              "prot_cls": pc_all,
                              "lig_xyz": lx,
                              "lig_cls": _lig_cls_from_x(np.asarray(lig["x"]))[: len(lx)]})
        msg = f"[voksel:{name}] {len(self.recs)} kompleks ({skipped} atlandi), kol={arm}"
        if clipped:
            msg += (f"  !! {clipped} komplekste ligand {GRID} A kutuya tam sigmiyor "
                    f"(bu komplekslerde kenar atomlari izgaraya girmez)")
        print(msg)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.savez_compressed(cache, recs=np.array(self.recs, dtype=object))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        px, pc, lx, lc = r["prot_xyz"], r["prot_cls"], r["lig_xyz"], r["lig_cls"]
        if self.arm == "vx_scr":                       # SIFIRDAN egitimde kalici hasar
            px, pc, lx, lc = perturb_coords(r, "scramble", salt="arm")
        if self.probe and self.probe != "none":        # cikarim aninda sonda
            rec = {"id": r["id"], "centre": r["centre"], "prot_xyz": px,
                   "prot_cls": pc, "lig_xyz": lx, "lig_cls": lc}
            px, pc, lx, lc = perturb_coords(rec, self.probe, salt="probe")
        return {"id": r["id"], "y": np.float32(r["y"]),
                "g": splat(px, pc, lx, lc, r["centre"])}


def vox_collate(items):
    return {"ids": [it["id"] for it in items],
            "g": torch.from_numpy(np.stack([it["g"] for it in items])),
            "y": torch.tensor([it["y"] for it in items], dtype=torch.float32)}


# --------------------------------------------------------------------------- model
class VoxelCNN(nn.Module):
    """Duz 3B evrisimli ag. Dikkat yok, mesafe kodlamasi yok, dil modeli yok.

    Cikis cift-gosterim modeliyle ayni: ortalama ve log varyans, beta-NLL ile
    egitilir, boylece iki mimarinin kayiplari ve belirsizlikleri karsilastirilabilir.
    """

    def __init__(self, width: int = 32, dropout: float = 0.15):
        super().__init__()
        w = width
        def blk(i, o, pool=True):
            layers = [nn.Conv3d(i, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool3d(2))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(blk(N_CH, w), blk(w, 2 * w), blk(2 * w, 4 * w),
                                 blk(4 * w, 4 * w, pool=False))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(8 * w, 4 * w), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(4 * w, 2))

    def forward(self, g):
        h = self.net(g)
        h = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1)
        o = self.head(self.drop(h))
        return o[:, 0], o[:, 1].clamp(-6.0, 4.0)


def beta_nll(mu, logvar, y, beta: float = 0.5):
    var = logvar.exp()
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)
    return (nll * var.detach() ** beta).mean()


# --------------------------------------------------------------------------- egitim
def run_epoch(model, loader, dev, opt=None, beta: float = 0.5):
    train = opt is not None
    model.train(train)
    tot, n, P, Y, I = 0.0, 0, [], [], []
    for b in loader:
        g, y = b["g"].to(dev, non_blocking=True), b["y"].to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            mu, lv = model(g)
            loss = beta_nll(mu, lv, y, beta)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot += float(loss.detach()) * len(y)
        n += len(y)
        P.append(mu.detach().float().cpu().numpy())
        Y.append(y.detach().float().cpu().numpy())
        I += b["ids"]
    return tot / max(n, 1), np.concatenate(P), np.concatenate(Y), I


def train_one(seed: int, tr, va, te, out: str, dev, epochs: int, bs: int, lr: float,
              width: int, workers: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VoxelCNN(width=width).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    kw = dict(num_workers=workers, collate_fn=vox_collate, pin_memory=(dev.type == "cuda"),
              persistent_workers=workers > 0)
    ltr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **kw)
    lva = DataLoader(va, batch_size=bs, **kw)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        t0 = time.time()
        trl, *_ = run_epoch(model, ltr, dev, opt)
        val, vp, vy, _ = run_epoch(model, lva, dev)
        vr = float(np.sqrt(((vp - vy) ** 2).mean()))
        sched.step(vr)
        if vr < best - 1e-4:
            best, bad = vr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [tohum {seed}] devir {ep + 1}/{epochs} egitim {trl:.4f} dogrulama RMSE {vr:.4f} "
              f"({time.time() - t0:.0f} sn)", flush=True)
        if bad >= 6:
            print(f"  [tohum {seed}] erken durdu")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    d = os.path.join(out, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    torch.save({"state": model.state_dict(), "width": width, "seed": seed, "n_param": npar},
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


# --------------------------------------------------------------------------- sonda bataryasi
DEFAULT_PROBES = ("none", "translate:0.5", "translate:1.0", "translate:2.0", "translate:4.0",
                  "translate:8.0", "rotate", "scale:1.5", "permute_res", "permute_atom", "scramble")
COST_THRESH, RBASE_THRESH = 0.05, 0.99


def probe_battery(out: str, ds, dev, probes=DEFAULT_PROBES, bs: int = 32, workers: int = 2,
                  tag: str = "test"):
    """Egitilmis her kontrol noktasini hasarli girdilerle yeniden degerlendirir."""
    seeds = sorted(d for d in os.listdir(out) if d.startswith("seed"))
    if not seeds:
        raise SystemExit(f"{out} altinda kontrol noktasi yok")
    rows, base = {}, {}
    for sd in seeds:
        ck = torch.load(os.path.join(out, sd, "model.pt"), map_location=dev, weights_only=False)
        model = VoxelCNN(width=ck["width"]).to(dev)
        model.load_state_dict(ck["state"])
        for spec in probes:
            ds.probe = spec
            _, p, y, _ = run_epoch(model, DataLoader(ds, batch_size=bs, num_workers=workers,
                                                     collate_fn=vox_collate), dev)
            m = regression_metrics(y, p)
            if spec == "none":
                base[sd] = p
                dmu, rb = 0.0, 1.0
            else:
                b = base[sd]
                dmu = float(np.abs(p - b).mean())
                rb = float(np.corrcoef(p, b)[0, 1]) if p.std() > 1e-9 and b.std() > 1e-9 else 0.0
            rows.setdefault(spec, []).append((m["rmse"], m["pearson"], dmu, rb))
            print(f"  [{sd}] {spec:16s} RMSE {m['rmse']:.4f}  d|mu| {dmu:.4f}  r_temel {rb:.4f}",
                  flush=True)
    ds.probe = "none"
    lines = [f"VOKSEL SONDA RAPORU ({tag}, n = {len(ds)}, {len(seeds)} model)", "=" * 62,
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
    uses = (cost > COST_THRESH) and (rb < RBASE_THRESH)
    lines += ["", f"geometriyi silmenin bedeli (scramble - none) = {cost:+.4f} pK",
              f"tohumlar arasi RMSE standart sapmasi            =  {agg['none']['rmse_sd']:.4f} pK",
              f"bozulmus tahminlerin korelasyonu                =  {rb:.4f}",
              f"KARAR: {'GEOMETRIYI KULLANIYOR' if uses else 'GEOMETRIYI KULLANMIYOR'} "
              f"(olcut: bedel > {COST_THRESH} pK VE r_temel < {RBASE_THRESH})"]
    txt = "\n".join(lines)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, f"probe_report_{tag}.txt"), "w").write(txt + "\n")
    json.dump(agg, open(os.path.join(out, f"probes_{tag}.json"), "w"), indent=2)
    print("\n" + txt)
    return agg


# --------------------------------------------------------------------------- CLI
def build_sets(a, arm: str):
    from .data import load_lp_pdbbind
    sp = load_lp_pdbbind(a.splits, clean_level=a.clean_level)
    mk = lambda df, nm: VoxelDataset(df, a.protein_pattern, a.ligand_pattern, cutoff=a.cutoff,
                                     max_res=a.max_res, name=nm, arm=arm,
                                     cache=(os.path.join(a.cache, f"{arm}_{nm}.npz") if a.cache else None))
    return mk(sp["train"], "train"), mk(sp["val"], "val"), mk(sp["test"], "test")


def main(argv=None):
    p = argparse.ArgumentParser(description="Voksel tabanli 3B-CNN kolu (ikinci mimari)")
    p.add_argument("--arm", default="vx", choices=["vx", "vx_scr", "vx_rand"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default="runs/vx")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--splits", default="LP_PDBBind.csv")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--protein_pattern", default="pdb/{id}/{id}_protein.pdb")
    p.add_argument("--ligand_pattern", default="pdb/{id}/{id}_ligand.sdf")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--max_res", type=int, default=64)
    p.add_argument("--cache", default="cache_vox")
    p.add_argument("--probes", default=",".join(DEFAULT_PROBES))
    p.add_argument("--probe-only", dest="probe_only", default=None,
                   help="egitimi atla, bu dizindeki kontrol noktalarina sonda uygula")
    a = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[voksel] cihaz = {dev}, izgara {GRID}^3 @ {SPACING} A, {N_CH} kanal")
    if dev.type != "cuda":
        print("[voksel] !! UYARI: GPU bulunamadi. CPU'da bu kol saatler surer.")

    if a.probe_only:
        _, _, te = build_sets(a, a.arm)
        probe_battery(a.probe_only, te, dev, probes=tuple(a.probes.split(",")),
                      workers=a.workers, tag="test")
        return

    tr, va, te = build_sets(a, a.arm)
    os.makedirs(a.out, exist_ok=True)
    res = [train_one(int(s), tr, va, te, a.out, dev, a.epochs, a.bs, a.lr, a.width, a.workers)
           for s in a.seeds.split(",")]
    r = np.array([x["test"]["rmse"] for x in res if "test" in x])
    print(f"\n[voksel:{a.arm}] test RMSE tohum basina {r.mean():.4f} +/- {r.std():.4f} ({len(r)} tohum)")
    json.dump({"arm": a.arm, "per_seed_rmse": r.tolist(), "mean": float(r.mean()),
               "sd": float(r.std())}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    if a.arm == "vx":
        probe_battery(a.out, te, dev, probes=tuple(a.probes.split(",")), workers=a.workers, tag="test")


if __name__ == "__main__":
    main()
