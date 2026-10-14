"""Voksel kolunun ucdan uca dumansiz calistigini sandbox'ta dogrular.

Gercek PDBbind verisi KULLANILMAZ; sentetik komplekslerle her kod yolu yurutulur.
Amac, kullanicinin Colab kosusunun bir NameError veya sekil hatasina carpmamasidir.
"""
import os, sys, tempfile, numpy as np, torch
sys.path.insert(0, "/home/claude/ensaff")
from ensaff.voxel import (GRID, SPACING, N_CH, N_RES_CLASS, N_LIG_CLASS, VoxelDataset,
                          VoxelCNN, splat, perturb_coords, vox_collate, train_one,
                          probe_battery, beta_nll, DEFAULT_PROBES)

rng = np.random.RandomState(0)
FAIL = []
def check(name, cond, extra=""):
    print(("  GECTI  " if cond else "  KALDI  ") + name + ("  " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


def make_recs(n=40, seed=0):
    r = np.random.RandomState(seed)
    out = []
    for i in range(n):
        npr, nl = r.randint(120, 300), r.randint(12, 30)
        centre = r.randn(3) * 3
        lig = (r.randn(nl, 3) * 2.0 + centre).astype(np.float32)
        prot = (r.randn(npr, 3) * 5.0 + centre).astype(np.float32)
        keep = (np.abs(prot - lig.mean(0)) <= GRID * SPACING / 2 - SPACING).all(1)
        prot = prot[keep]
        pc = r.randint(0, N_RES_CLASS, len(prot))
        lc = r.randint(0, N_LIG_CLASS, nl)
        # hedef, geometriye BAGLI olsun ki egitimin ogrenecek bir seyi olsun
        y = float(3.0 + 0.02 * np.sqrt(((prot[:, None] - lig[None]) ** 2).sum(-1)).min(1).mean()
                  + 0.1 * r.randn())
        out.append({"id": f"c{i:03d}", "y": y, "centre": lig.mean(0).astype(np.float32),
                    "prot_xyz": prot, "prot_cls": pc, "lig_xyz": lig, "lig_cls": lc})
    return out


def ds_from(recs, arm="vx"):
    d = VoxelDataset.__new__(VoxelDataset)
    d.recs, d.arm, d.probe = recs, arm, "none"
    return d


print("1. izgara (splat)")
recs = make_recs(6)
r0 = recs[0]
g = splat(r0["prot_xyz"], r0["prot_cls"], r0["lig_xyz"], r0["lig_cls"], r0["centre"])
check("sekil", g.shape == (N_CH, GRID, GRID, GRID), str(g.shape))
check("kutle korunuyor", abs(g.sum() - (len(r0["prot_xyz"]) + len(r0["lig_xyz"]))) < 1e-3,
      f"{g.sum():.4f} / {len(r0['prot_xyz']) + len(r0['lig_xyz'])}")
check("kanal kutleleri dogru",
      all(abs(g[k].sum() - (r0["prot_cls"] == k).sum()) < 1e-3 for k in range(N_RES_CLASS)))
check("negatif deger yok", (g >= 0).all())
# bir voksel altindaki kayma GORUNUR olmali (trilineer dagitimin tum amaci)
sub = splat(r0["prot_xyz"], r0["prot_cls"], r0["lig_xyz"] + np.float32([0.3, 0, 0]), r0["lig_cls"], r0["centre"])
check("0.3 A kayma izgarayi degistiriyor", np.abs(sub - g).sum() > 1e-3,
      f"L1 = {np.abs(sub - g).sum():.3f}")

print("\n2. sondalar (koordinat uzayi)")
for spec in DEFAULT_PROBES:
    px, pc, lx, lc = perturb_coords(r0, spec, salt="probe")
    px2, pc2, lx2, lc2 = perturb_coords(r0, spec, salt="probe")
    ok = np.array_equal(px, px2) and np.array_equal(lx, lx2) and np.array_equal(pc, pc2)
    check(f"{spec:16s} tekrarlanabilir", ok)
    if spec not in ("none",):
        gg = splat(px, pc, lx, lc, r0["centre"])
        moved = np.abs(gg - g).sum() > 1e-6
        check(f"{spec:16s} izgarayi degistiriyor", moved)
check("scramble bilesimi koruyor",
      abs(splat(*perturb_coords(r0, "scramble", salt="probe"), r0["centre"]).sum() - g.sum()) < 1e-3,
      f"{splat(*perturb_coords(r0, 'scramble', salt='probe'), r0['centre']).sum():.4f} / {g.sum():.4f}")
pxp, pcp, _, _ = perturb_coords(r0, "permute_res", salt="probe")
check("permute_res konumlari korur", np.array_equal(pxp, r0["prot_xyz"]))
check("permute_res sinif coklugunu korur",
      np.array_equal(np.bincount(pcp, minlength=N_RES_CLASS),
                     np.bincount(r0["prot_cls"], minlength=N_RES_CLASS)))
_, _, lxr, _ = perturb_coords(r0, "rotate", salt="probe")
check("rotate agirlik merkezini korur",
      np.allclose(lxr.mean(0), r0["lig_xyz"].mean(0), atol=1e-4))
_, _, lxt, _ = perturb_coords(r0, "translate:2.0", salt="probe")
check("translate kati (ic mesafeler sabit)",
      np.allclose(np.linalg.norm(lxt[1:] - lxt[:1], axis=1),
                  np.linalg.norm(r0["lig_xyz"][1:] - r0["lig_xyz"][:1], axis=1), atol=1e-4))

print("\n3. vx_scr kolu (sifirdan hasarli egitim)")
d_real, d_scr = ds_from(make_recs(4), "vx"), ds_from(make_recs(4), "vx_scr")
a, b = d_real[0]["g"], d_scr[0]["g"]
check("karistirilmis kol farkli izgara veriyor", np.abs(a - b).sum() > 1.0)
check("karistirilmis kol ayni kutleyi veriyor", abs(a.sum() - b.sum()) < 1e-3,
      f"{a.sum():.3f} / {b.sum():.3f}")
check("vx_scr devirler arasi sabit", np.array_equal(d_scr[0]["g"], d_scr[0]["g"]))

print("\n4. model ve kayip")
m = VoxelCNN(width=8)
batch = vox_collate([d_real[i] for i in range(4)])
mu, lv = m(batch["g"])
check("cikti sekli", mu.shape == (4,) and lv.shape == (4,))
loss = beta_nll(mu, lv, batch["y"])
loss.backward()
gn = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
check("gradyan akiyor", np.isfinite(float(loss)) and gn > 0, f"kayip {float(loss):.4f}")
check("parametre sayisi makul", 1e5 < sum(p.numel() for p in VoxelCNN(32).parameters()) < 5e6,
      f"{sum(p.numel() for p in VoxelCNN(32).parameters()) / 1e6:.3f} M")

print("\n5. ucdan uca egitim + sonda bataryasi")
tr, va, te = ds_from(make_recs(48, 1)), ds_from(make_recs(16, 2)), ds_from(make_recs(16, 3))
with tempfile.TemporaryDirectory() as out:
    res = train_one(0, tr, va, te, out, torch.device("cpu"), epochs=2, bs=8, lr=3e-3,
                    width=8, workers=0)
    check("egitim metrik dondurdu", "test" in res and np.isfinite(res["test"]["rmse"]))
    check("kontrol noktasi yazildi", os.path.exists(os.path.join(out, "seed0", "model.pt")))
    check("tahminler yazildi", os.path.exists(os.path.join(out, "seed0", "pred_test.npz")))
    agg = probe_battery(out, te, torch.device("cpu"),
                        probes=("none", "translate:2.0", "permute_res", "scramble"),
                        workers=0, tag="test")
    check("sonda raporu yazildi", os.path.exists(os.path.join(out, "probe_report_test.txt")))
    check("none sondasi degisim uretmiyor", abs(agg["none"]["dmu"]) < 1e-9)
    check("scramble sondasi tahminleri oynatiyor", agg["scramble"]["dmu"] > 0)

print("\n" + ("TUM TESTLER GECTI" if not FAIL else f"KALAN: {FAIL}"))
sys.exit(1 if FAIL else 0)
