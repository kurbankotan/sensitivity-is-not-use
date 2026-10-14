"""ccdyn'i sentetik veriyle ucdan uca yurutur. Gercek PDBbind KULLANILMAZ."""
import sys, os, tempfile, numpy as np
sys.path.insert(0, "/home/claude/ensaff")
from ensaff.ccdyn import (counts, edges_for, featurize, fit_predict, paired_bootstrap,
                          compare_to, N_RES_CLASS, N_LIG_CLASS)

FAIL = []
def ck(n, c, x=""):
    print(("  GECTI  " if c else "  KALDI  ") + n + ("  " + x if x else ""))
    if not c: FAIL.append(n)

r = np.random.RandomState(0)
P, N = 30, 15
D = (r.rand(P, N) * 14).astype(np.float32)
rc = r.randint(0, N_RES_CLASS, P)
lc = r.randint(0, N_LIG_CLASS, N)

print("1. kabuk merdiveni")
for ns in (0, 1, 2, 3, 5):
    f = counts(D, rc, lc, edges_for(ns))
    exp = N_RES_CLASS * N_LIG_CLASS * max(ns, 1)
    ck(f"{ns} kabuk -> {exp} ozellik", f.shape == (exp,), str(f.shape))
f0 = counts(D, rc, lc, edges_for(0))
ck("0 kabuk tum ciftleri sayiyor", abs(f0.sum() - P * N) < 1e-3, f"{f0.sum():.1f} / {P*N}")
f5 = counts(D, rc, lc, edges_for(5))
ck("5 kabuk yalnizca kesme icini sayiyor", f5.sum() <= P * N and f5.sum() == (D < 12.0).sum(),
   f"{f5.sum():.0f} / {(D<12.0).sum()}")
ck("0 kabuk = bilesimlerin dis carpimi",
   np.allclose(f0.reshape(N_RES_CLASS, N_LIG_CLASS),
               np.outer(np.bincount(rc, minlength=N_RES_CLASS),
                        np.bincount(lc, minlength=N_LIG_CLASS))))
w = r.rand(P).astype(np.float32)
fw = counts(D, rc, lc, edges_for(0), weight=w)
ck("agirlikli sayim agirliklari topluyor", abs(fw.sum() - w.sum() * N) < 1e-2)

print("\n2. ozellik matrisleri")
def rec(i, K):
    Dk = np.stack([(r.rand(P, N) * 14).astype(np.float32) for _ in range(K)])
    return {"id": f"c{i:03d}", "y": float(3 + r.randn()), "rc": rc, "lc": lc,
            "D": Dk[0], "Dk": Dk, "n_sib": K - 1,
            "rmsf": r.randn(P).astype(np.float32) if K > 1 else None}
recs = [rec(i, 1 + (i % 3)) for i in range(60)]
for agg, mult in (("ref", 1), ("mean", 1), ("meanstd", 2)):
    X, y, ids = featurize(recs, n_shell=5, agg=agg)
    ck(f"agg={agg:8s} -> {125*mult} ozellik", X.shape == (60, 125 * mult), str(X.shape))
single = [q for q in recs if q["Dk"].shape[0] == 1]
Xr, _, _ = featurize(single, n_shell=5, agg="ref")
Xm, _, _ = featurize(single, n_shell=5, agg="mean")
ck("K=1'de mean ile ref BIREBIR ayni", np.allclose(Xr, Xm), f"{len(single)} kayit")
X, y, ids = featurize(recs, n_shell=5, agg="ref", use_rmsf=True)
ck("rmsf bloku ekleniyor", X.shape == (60, 250), str(X.shape))
ck("rmsf'siz kayitlar sifirla dolduruluyor", np.isfinite(X).all())

print("\n3. model ve eslesmis onyukleme")
Xtr, ytr, _ = featurize(recs[:40], n_shell=5, agg="ref")
Xte, yte, ite = featurize(recs[40:], n_shell=5, agg="ref")
Pm = fit_predict(Xtr, ytr, {"test": (Xte, yte, ite)}, seeds=(0, 1), n_estimators=30)["test"]
ck("tahmin sekli", Pm.shape == (2, 20), str(Pm.shape))
d = paired_bootstrap(yte, Pm[0], Pm[1], n_boot=200)
ck("onyukleme aralik uretiyor", d["ci95"][0] <= d["diff"] <= d["ci95"][1], str(d))
same = paired_bootstrap(yte, Pm[0], Pm[0], n_boot=200)
ck("ayni tahmin -> sifir fark", abs(same["diff"]) < 1e-9 and not same["significant"])

print("\n4. sinir agi karsilastirmasi (csv ve npz)")
with tempfile.TemporaryDirectory() as t:
    import pandas as pd
    pd.DataFrame({"id": ite, "y": yte, "mu": Pm[0] + 0.3}).to_csv(f"{t}/pred_test.csv", index=False)
    np.savez(f"{t}/pred_test.npz", ids=ite, y=yte, p=Pm[0] + 0.3)
    out = []
    for f in ("pred_test.csv", "pred_test.npz"):
        d = compare_to(ite, yte, Pm[0], f"{t}/{f}", f, out.append, min_common=5)
        ck(f"{f} okundu ve karsilastirildi", d is not None and d["diff"] < 0,
           "" if d is None else f"fark {d['diff']:+.3f}")
    pd.DataFrame({"id": ite[:3], "y": yte[:3], "mu": Pm[0][:3]}).to_csv(f"{t}/az.csv", index=False)
    ck("ortak kimlik azsa None donuyor",
       compare_to(ite, yte, Pm[0], f"{t}/az.csv", "az", out.append, min_common=50) is None)
    ck("olmayan dosya zarifce atlaniyor",
       compare_to(ite, yte, Pm[0], f"{t}/yok.csv", "yok", out.append, min_common=5) is None)

print("\n" + ("TUM TESTLER GECTI" if not FAIL else f"KALAN: {FAIL}"))
sys.exit(1 if FAIL else 0)
