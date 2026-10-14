"""The consolidated report block printed at the end of the Colab pipeline.

Kept as a file here so it can be unit-tested in the sandbox against real metrics.json
output before being embedded into the notebook.
"""
import glob
import json
import os
import platform
from datetime import datetime


def _fmt(v, n=3):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def build_report(build_id: str) -> str:
    L = []
    A = L.append
    A("=" * 68)
    A(f"ENSAFF RAPOR | BUILD {build_id} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("=" * 68)

    # --- ortam
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        A(f"ORTAM: {gpu} | torch {torch.__version__} | python {platform.python_version()}")
    except Exception as e:
        A(f"ORTAM: okunamadi ({e})")
    try:
        import subprocess
        ram = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
        A(f"RAM: toplam {ram[1]} GB")
    except Exception:
        pass
    A("")

    files = sorted(glob.glob("runs/*/metrics.json"))
    if not files:
        A("!! runs/*/metrics.json bulunamadi - pipeline calismadi.")
        return "\n".join(L)

    # --- her kosunun config + metrikleri
    for f in files:
        run = f.split("/")[1]
        m = json.load(open(f))
        A("-" * 68)
        A(f"[{run}]")
        cfg = m.get("config")
        A(f"  config: {json.dumps(cfg, ensure_ascii=False)}" if cfg else "  config: YOK (eski kod calismis olabilir)")
        if "delta" in str(m):             # eslestirilmis karsilastirma raporu
            cfgc = m.get("config", {})
            A(f"  (fark = B - A, ortak alt kumede; A={cfgc.get('a')} B={cfgc.get('b')})")
            for k, v in m.items():
                if not isinstance(v, dict) or "delta" not in v:
                    continue
                A(f"  {k:<12} ortak n={v['n_common']} (A:{v['n_a']} B:{v['n_b']})")
                for mk in ("rmse", "pearson", "spearman"):
                    d = v["delta"][mk]
                    A(f"      {mk:<9} A={v['A'][mk]:+.3f} B={v['B'][mk]:+.3f} "
                      f"fark={d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                      f"{'ANLAMLI' if d['significant'] else 'fark yok'}")
            continue
        if "rho_model_sd" in str(m):      # applicability raporu: belirsizlik sinyallerinin karsilastirmasi
            A("  (belirsizlik sinyali ~ |hata| Spearman; buyuk ve pozitif = iyi)")
            for k, v in m.items():
                if not isinstance(v, dict) or "rho_model_sd" not in v:
                    continue
                A(f"  {k:<12} n={v['n']:<5} model_sd={v['rho_model_sd']:+.3f}  "
                  f"ligand_uzaklik={v['rho_lig_dist']:+.3f}  protein_uzaklik={v['rho_prot_dist']:+.3f}  "
                  f"ikisi={v['rho_both_dist']:+.3f}")
                if v.get("rmse_nearest_quartile") is not None:
                    A(f"               RMSE en yakin %25={v['rmse_nearest_quartile']:.3f} "
                      f"| en uzak %25={v['rmse_furthest_quartile']:.3f}"
                      + (f" | kapsama(mesafe)={v['coverage_distance']:.3f}" if "coverage_distance" in v else ""))
            continue
        if m.get("demo"):
            A(f"  NOT: {m.get('note', 'demo')}")
        if "conformal_q" in m:
            A(f"  conformal_q: {_fmt(m['conformal_q'])}")
        for key, sub in m.items():
            if not isinstance(sub, dict) or key == "config":
                continue
            inner = [sub] if "rmse" in sub else [v for v in sub.values() if isinstance(v, dict) and "rmse" in v]
            names = [key] if "rmse" in sub else [f"{key}:{k}" for k, v in sub.items() if isinstance(v, dict) and "rmse" in v]
            for nm, r in zip(names, inner):
                A(f"  {nm:<22} n={r.get('n','-'):<6} rmse={_fmt(r.get('rmse'))} "
                  f"r={_fmt(r.get('pearson'))} rho={_fmt(r.get('spearman'))} "
                  f"cov90={_fmt(r.get('coverage@90'))} w={_fmt(r.get('mean_interval_width'))} "
                  f"sd~err={_fmt(r.get('spearman_sd_vs_err'))}")

    # --- cok tohumlu kosularin ortalama +- std ozeti (seq_s0, seq_s1, ... / baseline_s0, ...)
    import re
    from collections import defaultdict

    def _splits_of(m):
        """metrics.json -> {etiket: metrik_dict}; hem duz hem (rf/xgb) ic ice yapiyi destekler."""
        out = {}
        for k, v in m.items():
            if not isinstance(v, dict) or k == "config":
                continue
            if "rmse" in v:
                out[k] = v
            else:
                for k2, v2 in v.items():
                    if isinstance(v2, dict) and "rmse" in v2:
                        out[f"{k}:{k2}"] = v2
        return out

    groups = defaultdict(list)
    for f in files:
        run = f.split("/")[1]
        base = re.sub(r"_s\d+$", "", run)
        if base != run:
            groups[base].append(f)
    for base, fs in sorted(groups.items()):
        if len(fs) < 2:
            continue
        A("-" * 68)
        A(f"[{base}] {len(fs)} tohum - ortalama +- std")
        per_label = defaultdict(lambda: defaultdict(list))
        for f in fs:
            for label, r in _splits_of(json.load(open(f))).items():
                for k in ("rmse", "pearson", "spearman", "coverage@90", "spearman_sd_vs_err"):
                    if r.get(k) is not None:
                        per_label[label][k].append(r[k])
        for label in sorted(per_label, key=lambda x: ("BDB" in x, "test" not in x, x)):
            parts = []
            for k in ("rmse", "pearson", "spearman", "coverage@90", "spearman_sd_vs_err"):
                vals = per_label[label][k]
                if vals:
                    mu = sum(vals) / len(vals)
                    sd = (sum((x - mu) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
                    parts.append(f"{k}={mu:.3f}+-{sd:.3f}")
            A(f"  {label:<16} " + "  ".join(parts))

    # --- geometri sondasi: metinler oldugu gibi aktarilir
    for pr in sorted(glob.glob("runs/*/probe_report.txt")):
        A("-" * 68)
        A(f"[{pr.split('/')[1]} geometri sondasi]")
        try:
            A("\n".join("  " + ln for ln in open(pr).read().splitlines()))
        except Exception as e:
            A(f"  okunamadi: {e}")

    # --- yorumlamayi gecersiz kilan durumlar; sonuclarin yaninda degil, tek yerde toplanir
    warn = []
    for f in files:
        run, m = f.split("/")[1], json.load(open(f))
        if m.get("conformal_degenerate"):
            warn.append(f"{run}: konformal q={_fmt(m.get('conformal_q'), 2)} dejenere -> cov90 ve "
                        f"aralik genisligi yorumlanamaz")
        for sp, c in (m.get("bfactor_degenerate") or {}).items():
            if c:
                warn.append(f"{run}/{sp}: {c} yapida B-faktor sutunu SABIT -> bu sette B-faktor "
                            f"karsilastirmasi gecersiz")
        for label, r in _splits_of(m).items():
            n = r.get("n")
            if isinstance(n, int) and n < 30:
                warn.append(f"{run}/{label}: n={n} -> iddia tasiyamaz, yalnizca betimleyici")
    if os.path.exists("runs/_notes.txt"):
        warn += [ln.strip() for ln in open("runs/_notes.txt").read().splitlines() if ln.strip()]
    if warn:
        A("-" * 68)
        A("!! YORUMLAMA UYARILARI")
        for w in sorted(set(warn)):
            A(f"  - {w}")

    # --- egitim egrisi ozeti
    A("-" * 68)
    for h in sorted(glob.glob("runs/*/history.csv")):
        try:
            import pandas as pd
            d = pd.read_csv(h)
            best = int(d["val_rmse"].idxmin())
            A(f"[{h.split('/')[1]} egitim] epoch={len(d)} en_iyi_epoch={best} "
              f"en_iyi_val_rmse={d['val_rmse'].min():.3f} son_val_rmse={d['val_rmse'].iloc[-1]:.3f} "
              f"tau={d['tau'].iloc[-1]:.3f} sn/epoch={d['sec'].mean():.0f}")
            A("  val_rmse: " + " ".join(f"{v:.3f}" for v in d["val_rmse"]))
            if "lr" in d.columns:
                A("  lr:       " + " ".join(f"{v:.1e}" for v in d["lr"]))
        except Exception as e:
            A(f"[{h}] okunamadi: {e}")
    A("=" * 68)
    return "\n".join(L)


def emit(build_id: str, work: str = None):
    """Print the report between copy markers, save it, and try to download it."""
    rep = build_report(build_id)
    print("\n\n>>>>>>>>>> BURADAN KOPYALA >>>>>>>>>>")
    print(rep)
    print("<<<<<<<<<< BURAYA KADAR <<<<<<<<<<\n")
    os.makedirs("results", exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(build_id))
    path = f"results/EnsAff_rapor_{safe}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(path, "w") as fh:
        fh.write(rep)
    print(f"[kaydedildi] {os.path.join(work or os.getcwd(), path)}")
    return path
