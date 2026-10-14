"""Voksel sonda bataryasinin POZITIF KONTROLU - sentetik veriyle, CPU'da.

Soru: bu batarya, geometriyi gercekten kullanan bir voksel modelini "kullaniyor"
diye isaretleyebiliyor mu? Isaretleyemiyorsa, gercek veride cikan "kullanmiyor"
karari hicbir sey demek degildir.

Iki sentetik rejim kurulur:

  GEOMETRIK   bilesim her komplekste BIREBIR AYNI tutulur (her siniftan ayni
              sayida atom), hedef yalnizca uzamsal duzenden gelir (4 A icindeki
              protein-ligand cifti sayisi). Konumu silen bir hasar hedefi tamamen
              silmek zorundadir.
  BILESIMSEL  geometri rastgeledir, hedef yalnizca atom sinifi sayimlarindan
              gelir. Konumu silen hasarin hicbir sey goturmemesi gerekir.

Beklenen: GEOMETRIK rejimde karar "KULLANIYOR", BILESIMSEL rejimde "KULLANMIYOR".
Ikisi de cikmazsa batarya ya kor ya da asiri duyarlidir.

CPU, yuksek RAM gerekmez, birkac dakika.
"""
import os, sys, tempfile, numpy as np, torch
sys.path.insert(0, "/home/claude/ensaff")
from ensaff.voxel import (GRID, SPACING, N_RES_CLASS, N_LIG_CLASS, VoxelDataset,
                          train_one, probe_battery)

LIM = GRID * SPACING / 2.0 - SPACING
N_PROT, N_LIG = 200, 20


def make(n, regime, seed):
    r = np.random.RandomState(seed)
    out = []
    for i in range(n):
        if regime == "geometric":
            # bilesim SABIT: her siniftan esit sayida atom
            pc = np.repeat(np.arange(N_RES_CLASS), N_PROT // N_RES_CLASS)
            lc = np.repeat(np.arange(N_LIG_CLASS), N_LIG // N_LIG_CLASS)
            lig = r.uniform(-4, 4, (N_LIG, 3)).astype(np.float32)
            prot = r.uniform(-LIM, LIM, (N_PROT, 3)).astype(np.float32)
            d = np.sqrt(((prot[:, None] - lig[None]) ** 2).sum(-1))
            y = 2.0 + 0.04 * float((d < 4.0).sum()) + 0.05 * r.randn()
        else:
            # geometri rastgele, hedef yalnizca sinif sayimlarindan
            pc = r.randint(0, N_RES_CLASS, N_PROT)
            lc = r.randint(0, N_LIG_CLASS, N_LIG)
            lig = r.uniform(-4, 4, (N_LIG, 3)).astype(np.float32)
            prot = r.uniform(-LIM, LIM, (N_PROT, 3)).astype(np.float32)
            y = 2.0 + 0.25 * float((lc == 0).sum()) + 0.02 * float((pc == 0).sum()) + 0.05 * r.randn()
        c = lig.mean(0).astype(np.float32)
        keep = (np.abs(prot - c) <= LIM).all(1)
        out.append({"id": f"{regime[:3]}{i:04d}", "y": float(y), "centre": c,
                    "prot_xyz": prot[keep], "prot_cls": pc[keep],
                    "lig_xyz": lig, "lig_cls": lc})
    return out


def ds(recs, arm="vx"):
    d = VoxelDataset.__new__(VoxelDataset)
    d.recs, d.arm, d.probe = recs, arm, "none"
    return d


SEEDS = (0, 1)
dev = torch.device("cpu")
report = {}
for regime in ("geometric", "compositional"):
    print(f"\n{'=' * 70}\nREJIM: {regime}\n{'=' * 70}")
    tr, va, te = make(400, regime, 1), make(120, regime, 2), make(120, regime, 3)
    arms = {"vx": [], "vx_scr": []}
    probes = []
    for sd in SEEDS:
        for arm in ("vx", "vx_scr"):
            with tempfile.TemporaryDirectory() as out:
                res = train_one(sd, ds(tr, arm), ds(va, arm), ds(te, arm), out, dev,
                                epochs=8, bs=16, lr=3e-3, width=10, workers=2)
                arms[arm].append(res["test"]["rmse"])
                if arm == "vx":
                    probes.append(probe_battery(out, ds(te, "vx"), dev,
                                                probes=("none", "translate:2.0", "permute_res", "scramble"),
                                                workers=2, tag=f"{regime}_s{sd}"))
    real, scr = np.array(arms["vx"]), np.array(arms["vx_scr"])
    cost_train = float(scr.mean() - real.mean())
    cost_probe = float(np.mean([p["scramble"]["rmse"] - p["none"]["rmse"] for p in probes]))
    rbase = float(np.mean([p["scramble"]["rbase"] for p in probes]))
    report[regime] = dict(real=float(real.mean()), scr=float(scr.mean()), cost_train=cost_train,
                          cost_probe=cost_probe, rbase=rbase,
                          real_sd=float(real.std()), scr_sd=float(scr.std()))
    print(f"\n  gercek kol test RMSE          {real.mean():.4f} +/- {real.std():.4f}")
    print(f"  karistirilmis kol test RMSE   {scr.mean():.4f} +/- {scr.std():.4f}")
    print(f"  SIFIRDAN EGITIM BEDELI        {cost_train:+.4f} pK   <- nedensel olcut")
    print(f"  CIKARIM SONDASI BEDELI        {cost_probe:+.4f} pK, r_temel {rbase:.4f}   <- duyarlilik olcutu")

print("\n" + "=" * 70 + "\nOZET\n" + "=" * 70)
g, c = report["geometric"], report["compositional"]
ok1 = g["cost_probe"] > 0.05 and g["rbase"] < 0.99
ok2 = not (c["cost_probe"] > 0.05 and c["rbase"] < 0.99)
print(f"GEOMETRIK  : karar {'KULLANIYOR' if ok1 else 'KULLANMIYOR'}   (beklenen KULLANIYOR)  "
      f"{'GECTI' if ok1 else 'KALDI'}")
print(f"BILESIMSEL : karar {'KULLANIYOR' if not ok2 else 'KULLANMIYOR'}   (beklenen KULLANMIYOR)  "
      f"{'GECTI' if ok2 else 'KALDI'}")
ok3 = g["cost_train"] > 0.05 and abs(c["cost_train"]) < 0.05
print(f"\nSIFIRDAN EGITIM KOLU (nedensel olcut)")
print(f"  geometrik  {g['cost_train']:+.4f} pK  (beklenen buyuk pozitif)")
print(f"  bilesimsel {c['cost_train']:+.4f} pK  (beklenen sifir)        {'GECTI' if ok3 else 'KALDI'}")
print(f"\nCIKARIM SONDASI (duyarlilik olcutu)")
print(f"  geometrik  {g['cost_probe']:+.4f} pK, r_temel {g['rbase']:.4f}")
print(f"  bilesimsel {c['cost_probe']:+.4f} pK, r_temel {c['rbase']:.4f}")
print(f"  oran (geometrik / bilesimsel) = {g['cost_probe'] / max(c['cost_probe'], 1e-9):.2f}x")
print("\nSONUC:")
print("  * Sifirdan egitim kolu her iki rejimde de DOGRU cevabi verdi." if ok3 else
      "  * Sifirdan egitim kolu yanildi - bu beklenmiyordu, incelenmeli.")
print("  * Cikarim sondasinin MUTLAK degeri bilesimsel rejimde YANLIS POZITIF verdi:"
      if not ok2 else "  * Cikarim sondasi her iki rejimde de dogru.")
if not ok2:
    print("    hasarli izgara modelin hic gormedigi bir dagilimdan gelir, bu yuzden")
    print("    tahmin bozulur; bozulma geometrinin KULLANILDIGINI degil girdinin")
    print("    DAGILIM DISI oldugunu gosterir.")
    print("  * Sondanin KARSILASTIRMALI okunmasi gecerli kalir: ayni hasara karsi")
    print(f"    geometriyi kullanan model {g['cost_probe'] / max(c['cost_probe'], 1e-9):.1f} kat daha duyarli.")
import json; json.dump(report, open("/tmp/claude-0/vxval_report.json", "w"), indent=2)
