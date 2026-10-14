"""Esdegisken kolu sentetik veriyle ucdan uca yurutur ve DEGISMEZLIGI dogrular."""
import sys, os, tempfile, numpy as np, torch
sys.path.insert(0, "/home/claude/ensaff")
from ensaff.equiv import (EquivDataset, EquivNet, collate, perturb, train_one,
                          probe_battery, N_RES_CLASS, N_LIG_CLASS, N_CLASS, RADIUS, _rand_rotation)
torch.set_num_threads(2)
FAIL = []
def ck(n, c, x=""):
    print(("  GECTI  " if c else "  KALDI  ") + n + ("  " + x if x else ""))
    if not c: FAIL.append(n)

r = np.random.RandomState(0)
def rec(i):
    P_, L_ = 40, 18
    c = r.randn(3).astype(np.float32) * 5
    pxyz = (r.randn(P_, 3) * 6 + c).astype(np.float32)
    lxyz = (r.randn(L_, 3) * 2 + c).astype(np.float32)
    xyz = np.concatenate([pxyz, lxyz], 0)
    cls = np.concatenate([r.randint(0, N_RES_CLASS, P_),
                          r.randint(0, N_LIG_CLASS, L_) + N_RES_CLASS])
    isl = np.concatenate([np.zeros(P_, np.float32), np.ones(L_, np.float32)])
    d = np.sqrt(((pxyz[:, None] - lxyz[None]) ** 2).sum(-1)).min(1)
    return {"id": f"c{i:03d}", "y": float(3 + 0.05 * d.mean() + 0.1 * r.randn()),
            "xyz": xyz, "cls": cls, "is_lig": isl, "centre": lxyz.mean(0).astype(np.float32)}

def ds_from(recs, arm="eq"):
    d = EquivDataset.__new__(EquivDataset); d.recs, d.arm, d.probe = recs, arm, "none"; return d

recs = [rec(i) for i in range(8)]
m = EquivNet(dim=32, layers=3).eval()
d0 = ds_from(recs)
b0 = collate([d0[i] for i in range(8)])
with torch.no_grad():
    mu0, _ = m(b0)

print("1. DEGISMEZLIK (mimarinin tanimlayici ozelligi)")
b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
b["xyz"] = b["xyz"] + torch.tensor([3.0, -7.0, 11.0])
with torch.no_grad(): mu, _ = m(b)
ck("tum kompleksi otelemek ciktiyi degistirmiyor", torch.allclose(mu, mu0, atol=2e-4),
   f"max fark {float((mu-mu0).abs().max()):.2e}")
R = torch.from_numpy(_rand_rotation(np.random.RandomState(1)))
b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
b["xyz"] = b["xyz"] @ R.T
with torch.no_grad(): mu, _ = m(b)
ck("tum kompleksi dondurmek ciktiyi degistirmiyor", torch.allclose(mu, mu0, atol=2e-4),
   f"max fark {float((mu-mu0).abs().max()):.2e}")

print("\n2. BAGIL geometri bozulunca cikti DEGISMELI")
for spec in ("translate:2.0", "rotate", "scale:1.5", "permute_res", "permute_atom", "scramble"):
    d0.probe = spec
    bb = collate([d0[i] for i in range(8)])
    with torch.no_grad(): mu, _ = m(bb)
    ck(f"{spec:14s} ciktiyi degistiriyor", not torch.allclose(mu, mu0, atol=1e-5),
       f"max fark {float((mu-mu0).abs().max()):.2e}")
d0.probe = "none"

print("\n3. hasar kollarinin degismezleri")
x0, c0, l0 = perturb(recs[0], "scramble", salt="arm")
ck("scramble dugum bilesimini koruyor",
   np.array_equal(np.bincount(c0, minlength=N_CLASS), np.bincount(recs[0]["cls"], minlength=N_CLASS)))
ck("scramble tekrarlanabilir", np.array_equal(x0, perturb(recs[0], "scramble", salt="arm")[0]))
_, cp, _ = perturb(recs[0], "permute_res", salt="probe")
ck("permute_res sinif coklugunu koruyor",
   np.array_equal(np.bincount(cp, minlength=N_CLASS), np.bincount(recs[0]["cls"], minlength=N_CLASS)))
ck("permute_res ligand etiketlerine dokunmuyor",
   np.array_equal(cp[recs[0]["is_lig"].astype(bool)], recs[0]["cls"][recs[0]["is_lig"].astype(bool)]))
dc = ds_from(recs, "eq_const")
bc = collate([dc[i] for i in range(8)])
ck("eq_const bayragi tasiniyor", bool(bc["const"].all()))
ck("eq_const koordinatlari DEGISTIRMIYOR (kenarlar gercek)",
   torch.allclose(bc["xyz"], b0["xyz"]))
with torch.no_grad(): muc, _ = m(bc)
ck("eq_const ciktiyi degistiriyor (olcu silindi)", not torch.allclose(muc, mu0, atol=1e-5))

print("\n4. ucdan uca egitim + sonda")
tr, va, te = ds_from([rec(i) for i in range(40)]), ds_from([rec(i) for i in range(40,52)]), \
             ds_from([rec(i) for i in range(52,64)])
with tempfile.TemporaryDirectory() as out:
    res = train_one(0, tr, va, te, out, torch.device("cpu"), epochs=2, bs=8, lr=3e-3,
                    dim=32, workers=0)
    ck("egitim metrik dondurdu", "test" in res and np.isfinite(res["test"]["rmse"]))
    agg = probe_battery(out, te, torch.device("cpu"),
                        probes=("none", "translate:2.0", "scramble"), workers=0, tag="test")
    ck("none sondasi degisim uretmiyor", abs(agg["none"]["dmu"]) < 1e-9)
    ck("scramble tahminleri oynatiyor", agg["scramble"]["dmu"] > 0)
print(f"\nparametre (dim=128): {sum(p.numel() for p in EquivNet(dim=128).parameters())/1e6:.3f} M")
print("\n" + ("TUM TESTLER GECTI" if not FAIL else f"KALAN: {FAIL}"))
sys.exit(1 if FAIL else 0)
