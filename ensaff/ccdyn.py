r"""Temas sayimi uzerinde: mesafe cozunurlugu, GERCEK konformasyon toplulugu ve hareketlilik.


NEDEN BU MODUL VAR
------------------
v14 temas-sayim modelinin geometriyi kullandigini ve sinir aglari kadar iyi
tahmin ettigini gosterdi. Bu, makalede iki boslugu acik birakti.

1) Sinir aglari icin "geometriyi kaldirip sifirdan egit" kolunu kosabildik ama
   temas sayimi icin kosmadik; sinirliliklara "mesafe olmadan hic ozelligi
   kalmaz" diye yazdim. Bu tam dogru degil. Kabuk SAYISINI dusurmek mesafe
   cozunurlugunu kademeli olarak siler: bes kabuk tam cozunurluk, tek kabuk
   yalnizca bir kesme yaricapi, sifir kabuk ise kesme bile yok (cepteki her
   cift bir kez sayilir), yani saf bilesim. Bu, o model icin birebir eslesen
   "sabit mesafe" koludur ve olculebilir.

2) Projenin ozgun sorusu protein hareketliliginin baglanma gucune katkisiydi.
   O soru simdiye kadar GEOMETRIYE KOR bir model uzerinde soruldu, bu yuzden
   cevabi yorumlanamazdi: geometrinin kendisini okumayan bir modele geometrinin
   DEGISIMINI sormak bos bir testtir. Temas-sayim modeli geometriyi kullandigi
   KANITLANMIS bir modeldir, dolayisiyla adil test burada yapilabilir.

Konformasyonlar sentetik degildir. `conformers.py` ayni proteinin baska PDB
kayitlarini cebe gore ust uste bindirir; ligand pozu sabit kalir, yalnizca
protein atomlarinin yeri degisir. Kardesler ayni bolme icinden secilir ve
yalnizca GEOMETRI katar, etiketleri hic okunmaz.

KOLLAR
------
  shells      kabuk sayisi 0,1,2,3,5 ile yeniden egitim (mesafe cozunurlugu)
  dyn         K konformasyon uzerinden toplama (ortalama; ortalama+sapma)
  rmsf        kardesler arasi kalinti hareketliligi ile agirliklandirilmis sayimlar

KULLANIM
--------
  python -m ensaff.ccdyn --arms shells,dyn,rmsf --out runs/ccdyn

CPU yeterlidir, GPU KULLANILMAZ, yuksek RAM gerekmez.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .conformers import ensemble_rmsf, pocket_states_from_siblings, sibling_index
from .contactcount import (LIG_CLASS, N_LIG_CLASS, N_RES_CLASS, RES_CLASS, _lig_classes)
from .data import load_external_sets, load_lp_pdbbind
from .evaluate import regression_metrics
from .features import extract_pocket, featurize_mol, load_ligand_poses, parse_pdb

DEFAULT_EDGES = (4.0, 6.0, 8.0, 10.0, 12.0)


def edges_for(n_shell: int):
    """Kabuk sayisindan kenarlar. 0 -> kesme yok (saf bilesim)."""
    if n_shell <= 0:
        return ()
    if n_shell == 1:
        return (12.0,)
    if n_shell == 2:
        return (6.0, 12.0)
    if n_shell == 3:
        return (4.0, 8.0, 12.0)
    if n_shell == 5:
        return DEFAULT_EDGES
    raise ValueError(f"desteklenmeyen kabuk sayisi: {n_shell}")


def counts(D, rc, lc, edges, weight=None):
    """[res_class, lig_class, shell] sayimlari, duzlestirilmis.

    edges bos ise mesafe hic kullanilmaz: cepteki her kalinti-atom cifti bir kez
    sayilir, yani ozellik vektoru iki bilesimin dis carpimidir. Geometrinin
    TAMAMEN kaldirildigi durum budur.
    `weight` verilirse her kalintinin katkisi o agirlikla carpilir (RMSF kolu).
    """
    ns = max(len(edges), 1)
    feat = np.zeros((N_RES_CLASS, N_LIG_CLASS, ns), dtype=np.float32)
    if not edges:
        ri, ai = np.meshgrid(np.arange(D.shape[0]), np.arange(D.shape[1]), indexing="ij")
        ri, ai = ri.ravel(), ai.ravel()
        w = np.ones(len(ri), dtype=np.float32) if weight is None else weight[ri]
        np.add.at(feat, (rc[ri], lc[ai], 0), w)
        return feat.reshape(-1)
    shell = np.searchsorted(np.asarray(edges), D, side="left")
    keep = shell < len(edges)
    if keep.any():
        ri, ai = np.nonzero(keep)
        w = np.ones(len(ri), dtype=np.float32) if weight is None else weight[ri]
        np.add.at(feat, (rc[ri], lc[ai], shell[ri, ai]), w)
    return feat.reshape(-1)


# --------------------------------------------------------------------------- kayit kurulumu
def build_records(df, pp_pat, lp_pat, name, cutoff=10.0, max_res=64, max_atoms=72,
                  max_poses=4, max_states=4, want_siblings=True, cache=None):
    """Kompleks basina: referans D, kalinti/atom siniflari, kardes konformasyonlari, RMSF.

    Kardesler YALNIZCA `df` icinden secilir, yani ayni bolmeden. Bir kardesin
    etiketi hicbir yerde okunmaz; katkisi sadece protein atomlarinin konumudur.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[ccdyn:{name}] {len(z['recs'])} kompleks (onbellekten)")
        return list(z["recs"])
    sib = sibling_index(df) if want_siblings else {}
    chains_cache, recs, skipped = {}, [], 0

    CACHE_MAX = 1500        # ayristirilmis yapilari sinirli tut: yuksek RAM gerekmesin
    def chains_of(cid):
        if cid not in chains_cache:
            if len(chains_cache) >= CACHE_MAX:
                chains_cache.pop(next(iter(chains_cache)))
            p = pp_pat.format(id=cid)
            chains_cache[cid] = parse_pdb(p) if os.path.exists(p) else None
        return chains_cache[cid]

    for r in df.itertuples():
        cid = str(r.id)
        pp, lp = pp_pat.format(id=cid), lp_pat.format(id=cid)
        if not (os.path.exists(pp) and os.path.exists(lp)):
            skipped += 1
            continue
        mol, poses = load_ligand_poses(lp, max_poses=max_poses)
        lig = featurize_mol(mol, max_atoms=max_atoms) if mol is not None else None
        if lig is None or not poses:
            skipped += 1
            continue
        chains = chains_of(cid)
        if chains is None:
            skipped += 1
            continue
        sel, D = extract_pocket(chains, poses, cutoff=cutoff, max_res=max_res)
        if not sel or D is None:
            skipped += 1
            continue
        letters = [chains[ci]["seq"][ri] for ci, ri in sel]
        rc = np.array([RES_CLASS.get(c, 0) for c in letters], dtype=np.int64)
        lc = _lig_classes(np.asarray(lig["x"]))
        sib_ids = [i for i in sib.get(str(r.seq), []) if i != cid][: max_states * 2]
        sib_chains = [c for c in (chains_of(i) for i in sib_ids) if c]
        Dk, used, rmsf = D[:1], 0, None
        if sib_chains:
            Dk, used = pocket_states_from_siblings(chains, sel, poses[0], sib_chains,
                                                   max_states=max_states)
            rmsf = ensemble_rmsf(chains, sel, sib_chains)
        recs.append({"id": cid, "y": float(r.y), "rc": rc, "lc": lc,
                     "D": D[0].astype(np.float32),          # referans poz
                     "Dk": Dk.astype(np.float32),           # [K, P, N] gercek konformasyonlar
                     "n_sib": int(used),
                     "rmsf": None if rmsf is None else rmsf[:, 0].astype(np.float32)})
    ks = np.array([r["Dk"].shape[0] for r in recs])
    print(f"[ccdyn:{name}] {len(recs)} kompleks ({skipped} atlandi)  "
          f"K dagilimi: {dict(zip(*np.unique(ks, return_counts=True)))}  "
          f"RMSF olculebilen: {sum(r['rmsf'] is not None for r in recs)}")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, recs=np.array(recs, dtype=object))
    return recs


# --------------------------------------------------------------------------- ozellik matrisleri
def featurize(recs, n_shell=5, agg="ref", use_rmsf=False, dup=False):
    """agg: ref | mean | meanstd ; use_rmsf: RMSF agirlikli blok ekle ;
    dup: ozellikleri ikiye katla (YENI BILGI YOK).

    `dup`, 250 ozellikli kollarin SEYRELME kontroludur: 125'ten 250'ye cikmanin
    kendisi ne kadar zarar veriyor? Bu olculmeden, 250 ozellikli bir kolun
    kotulesmesi bilgiye mi boyuta mi bagli bilinemez.
    """
    edges = edges_for(n_shell)
    X, y, ids = [], [], []
    for r in recs:
        if agg == "ref":
            f = counts(r["D"], r["rc"], r["lc"], edges)
        else:
            per = np.stack([counts(Dk, r["rc"], r["lc"], edges) for Dk in r["Dk"]])
            f = per.mean(0) if agg == "mean" else np.concatenate([per.mean(0), per.std(0)])
        if use_rmsf:
            w = r["rmsf"] if r["rmsf"] is not None else np.zeros(len(r["rc"]), dtype=np.float32)
            f = np.concatenate([f, counts(r["D"], r["rc"], r["lc"], edges, weight=w)])
        if dup:
            f = np.concatenate([f, f])
        X.append(f)
        y.append(r["y"])
        ids.append(r["id"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.array(ids)


# --------------------------------------------------------------------------- model ve karsilastirma
def fit_predict(Xtr, ytr, sets, seeds=(0, 1, 2, 3, 4), n_estimators=500):
    """Rastgele orman, tohum basina; her bolme icin topluluk ve tohum tahminleri."""
    from sklearn.ensemble import RandomForestRegressor
    out = {k: [] for k in sets}
    for sd in seeds:
        m = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=sd,
                                  min_samples_leaf=2)
        m.fit(Xtr, ytr)
        for k, (X, _y, _i) in sets.items():
            out[k].append(m.predict(X))
    return {k: np.stack(v) for k, v in out.items()}       # [tohum, n]


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=0):
    """(B - A) icin %95 aralik; iki model ayni yeniden ornekleme uzerinde."""
    rng = np.random.RandomState(seed)
    n, d = len(y), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        d.append(float(np.sqrt(((pb[i] - y[i]) ** 2).mean()) - np.sqrt(((pa[i] - y[i]) ** 2).mean())))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


def load_network_preds(path):
    """Sinir agi tahminlerini oku: pred_*.csv (cift gosterim) veya pred_*.npz (voksel)."""
    if path.endswith(".csv"):
        d = pd.read_csv(path)
        col = "mu" if "mu" in d.columns else ("pred" if "pred" in d.columns else d.columns[-1])
        idc = "id" if "id" in d.columns else d.columns[0]
        yc = "y" if "y" in d.columns else None
        return (d[idc].astype(str).values, d[yc].values if yc else None, d[col].values)
    z = np.load(path, allow_pickle=True)
    return z["ids"].astype(str), z["y"], z["p"]


def compare_to(ids, y, p, path, label, log, min_common: int = 50):
    """Temas sayimini bir sinir agiyla ortak kimlikler uzerinde eslesmis olarak karsilastirir."""
    if not os.path.exists(path):
        log(f"  {label:34s} tahmin dosyasi yok ({path})")
        return None
    nid, ny, npred = load_network_preds(path)
    common = np.intersect1d(ids, nid)
    if len(common) < min_common:
        log(f"  {label:34s} ortak kimlik az ({len(common)})")
        return None
    a = {i: v for i, v in zip(nid, npred)}
    b = {i: v for i, v in zip(ids, p)}
    yy = {i: v for i, v in zip(ids, y)}
    cy = np.array([yy[i] for i in common])
    net = np.array([a[i] for i in common])
    cc = np.array([b[i] for i in common])
    d = paired_bootstrap(cy, net, cc)          # B - A = temas sayimi - sinir agi
    log(f"  {label:34s} n={len(common):5d}  sinir agi {np.sqrt(((net-cy)**2).mean()):.4f}  "
        f"temas {np.sqrt(((cc-cy)**2).mean()):.4f}  fark {d['diff']:+.4f} "
        f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
        f"{'ANLAMLI' if d['significant'] else 'anlamli degil'}")
    return d


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="Temas sayimi: cozunurluk, konformasyon, hareketlilik")
    p.add_argument("--arms", default="shells,dyn,rmsf")
    p.add_argument("--shells", default="0,1,2,3,5")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out", default="runs/ccdyn")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--protein_pattern", default="pdbstruct/{id}/protein.pdb")
    p.add_argument("--ligand_pattern", default="pdbstruct/{id}/ligand.sdf")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--max_res", type=int, default=64)
    p.add_argument("--max_states", type=int, default=4)
    p.add_argument("--cache", default="cache_ccdyn")
    p.add_argument("--net_pair", default="runs/struct_ens/pred_test.csv")
    p.add_argument("--net_voxel", default="runs/vx/seed0/pred_test.npz")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    L = []
    def log(s=""):
        print(s, flush=True)
        L.append(s)

    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=a.clean_level)
    mk = lambda df, n: build_records(df, a.protein_pattern, a.ligand_pattern, n, cutoff=a.cutoff,
                                     max_res=a.max_res, max_states=a.max_states,
                                     cache=(os.path.join(a.cache, f"{n}.npz") if a.cache else None))
    tr, te = mk(tabs["train"], "train"), mk(tabs["test"], "test")
    seeds = tuple(int(x) for x in a.seeds.split(","))
    arms = set(a.arms.split(","))
    res = {}

    log("=" * 78)
    log("TEMAS SAYIMI UZERINDE COZUNURLUK, KONFORMASYON VE HAREKETLILIK")
    log("=" * 78)
    ks = np.array([r["Dk"].shape[0] for r in te])
    log(f"test bolmesinde konformasyon sayisi K: ortalama {ks.mean():.2f}, "
        f"K>1 olan kompleks {int((ks > 1).sum())}/{len(ks)} "
        f"({100 * (ks > 1).mean():.0f}%)")
    log(f"RMSF olculebilen test kompleksi: {sum(r['rmsf'] is not None for r in te)}/{len(te)}")

    def run(tag, **kw):
        Xtr, ytr, _ = featurize(tr, **kw)
        Xte, yte, ite = featurize(te, **kw)
        P = fit_predict(Xtr, ytr, {"test": (Xte, yte, ite)}, seeds=seeds)["test"]
        ens = P.mean(0)
        per = np.array([float(np.sqrt(((q - yte) ** 2).mean())) for q in P])
        m = regression_metrics(yte, ens)
        res[tag] = {"kw": kw, "ensemble_rmse": m["rmse"], "pearson": m["pearson"],
                    "per_seed_mean": float(per.mean()), "per_seed_sd": float(per.std(ddof=1)),
                    "n_feat": int(Xtr.shape[1]), "n": int(len(yte))}
        np.savez(os.path.join(a.out, f"pred_{tag}.npz"), ids=ite, y=yte, p=ens, per_seed=P)
        log(f"  {tag:22s} ozellik {Xtr.shape[1]:4d}  topluluk RMSE {m['rmse']:.4f}"
            f"  Pearson {m['pearson']:.4f}  tohum basina {per.mean():.4f} +/- {per.std(ddof=1):.4f}")
        return ite, yte, ens

    if "shells" in arms:
        log("\n[A] MESAFE COZUNURLUGU (sifirdan egitim, kabuk sayisi azaltilarak)")
        log("  0 kabuk = kesme bile yok, saf bilesim; geometrinin TAMAMEN kaldirildigi kol")
        base = None
        for ns in [int(x) for x in a.shells.split(",")]:
            ite, yte, ens = run(f"shell{ns}", n_shell=ns, agg="ref")
            if ns == 5:
                base = (ite, yte, ens)
        if base is not None:
            log("\n  5 kabuga karsi eslesmis farklar (pozitif = az kabuk daha kotu):")
            for ns in [int(x) for x in a.shells.split(",")]:
                if ns == 5:
                    continue
                z = np.load(os.path.join(a.out, f"pred_shell{ns}.npz"), allow_pickle=True)
                d = paired_bootstrap(base[1], base[2], z["p"])
                res[f"shell{ns}_vs_5"] = d
                log(f"    {ns} kabuk: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
                    f"{'ANLAMLI' if d['significant'] else 'anlamli degil'}")

    if "dyn" in arms:
        log("\n[B] GERCEK KONFORMASYON TOPLULUGU (kardes kristal yapilar)")
        ite, yte, ref = run("dyn_ref", n_shell=5, agg="ref")
        for agg in ("mean", "meanstd"):
            _, _, q = run(f"dyn_{agg}", n_shell=5, agg=agg)
            d = paired_bootstrap(yte, ref, q)
            res[f"dyn_{agg}_vs_ref"] = d
            log(f"    {agg:8s} referansa karsi: {d['diff']:+.4f} "
                f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
                f"{'ANLAMLI' if d['significant'] else 'anlamli degil'}")

    if "rmsf" in arms:
        log("\n[C] HAREKETLILIK OZELLIKLERI (kardesler arasi RMSF ile agirlikli sayimlar)")
        ite, yte, ref = run("rmsf_off", n_shell=5, agg="ref")
        _, _, q = run("rmsf_on", n_shell=5, agg="ref", use_rmsf=True)
        d = paired_bootstrap(yte, ref, q)
        res["rmsf_vs_off"] = d
        log(f"    RMSF eklendiginde: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
            f"{'ANLAMLI' if d['significant'] else 'anlamli degil'}")
        # SEYRELME KONTROLU: 250 ozellige cikmanin kendisi ne kadar zarar veriyor
        _, _, qd = run("dup_control", n_shell=5, agg="ref", dup=True)
        dd = paired_bootstrap(yte, ref, qd)
        res["dup_vs_off"] = dd
        log(f"    yalnizca ozellik ikiye katlanirsa (yeni bilgi yok): {dd['diff']:+.4f} "
            f"[{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  "
            f"{'ANLAMLI' if dd['significant'] else 'anlamli degil'}")
        log(f"    -> RMSF'in seyrelmeyi asan katkisi: "
            f"{d['diff'] - dd['diff']:+.4f} pK")

    log("\n[D] SINIR AGLARINA KARSI ESLESMIS KARSILASTIRMA (5 kabuk, referans poz)")
    base_npz = os.path.join(a.out, "pred_shell5.npz")
    if not os.path.exists(base_npz):
        base_npz = os.path.join(a.out, "pred_dyn_ref.npz")
    if os.path.exists(base_npz):
        z = np.load(base_npz, allow_pickle=True)
        for path, lab in ((a.net_pair, "cift gosterim (pair)"), (a.net_voxel, "voksel 3B-CNN")):
            d = compare_to(z["ids"].astype(str), z["y"], z["p"], path, lab, log)
            if d:
                res[f"vs_{lab.split()[0]}"] = d
    else:
        log("  temel kol kosulmadi, karsilastirma atlandi")

    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(L) + "\n")
    log(f"\nyazildi: {a.out}/report.txt")


if __name__ == "__main__":
    main()
