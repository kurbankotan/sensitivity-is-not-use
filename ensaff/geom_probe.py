r"""Does the structure model actually read the pocket geometry?

Every dynamics result so far is null: the K-state aggregation is identical to three decimals for K=1,3,5,9,
and the B-factor flexibility features do not replicate over five seeds. Before blaming the dynamics idea it
is worth asking whether the model is sensitive to *static* geometry at all, because an 0.8 angstrom rigid
displacement of the ligand already changes the RBF distance encoding by about 65% of its own norm while the
prediction moves by ~0.001 pK. A model that ignores the geometry it is given cannot be helped by giving it
better geometry.

This re-evaluates trained checkpoints under controlled damage to the residue-atom distance channel and
reports, next to the usual accuracy, two numbers that answer the question directly:

    d|mu|   mean absolute change in the predicted pK against the unperturbed run
    r_base  Pearson correlation between perturbed and unperturbed predictions

r_base near 1.000 with d|mu| near 0 means the channel is being ignored. The probes run in increasing order
of damage, ending with ones that no real structure could produce (const, permute), so the ceiling of the
effect is bracketed rather than guessed.

    python -m ensaff.geom_probe --ckpts 'runs/struct_s*/best.pt' --out runs/probe_struct

GPU, high-RAM, inference only - no training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .data import StructAffinityDataset, collate, load_external_sets, load_lp_pdbbind
from .embed import ProteinEmbedder
from .evaluate import regression_metrics
from .model import EnsAff
from .train import predict

DEFAULT_PROBES = ("none,translate:0.5,translate:1.0,translate:2.0,translate:4.0,translate:8.0,"
                  "rotate,scale:1.5,permute_res,permute_atom,const:8")


def _ds_key(args):
    """Dataset-relevant settings. Checkpoints that agree on these can share one built dataset."""
    return (args["clean_level"], args["cutoff"], args["max_res"], args["max_atoms"], args["max_poses"],
            bool(args.get("use_bfactor", False)), args.get("pocket_mode", "nearest"),
            args.get("n_true", 48), args.get("geom", "real"),
            args["protein_pattern"], args["ligand_pattern"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True, help="glob or comma-separated list of best.pt files")
    p.add_argument("--out", default="runs/probe")
    p.add_argument("--probes", default=DEFAULT_PROBES)
    p.add_argument("--splits", default="test,BDB2020+")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--cache_dir", default="")
    a = p.parse_args(argv)

    cks = sorted(glob.glob(a.ckpts)) if any(c in a.ckpts for c in "*?[") else \
        [c.strip() for c in a.ckpts.split(",") if c.strip()]
    cks = [c for c in cks if os.path.exists(c)]
    if not cks:
        print(f"[probe] kontrol noktasi bulunamadi: {a.ckpts}")
        return None
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probes = [s.strip() for s in a.probes.split(",") if s.strip()]
    if probes[0] != "none":
        probes = ["none"] + [x for x in probes if x != "none"]   # d|mu| and r_base need the baseline first
    want = [s.strip() for s in a.splits.split(",") if s.strip()]

    loaded = []
    for c in cks:
        ck = torch.load(c, weights_only=False)
        if ck["args"].get("mode") != "struct":
            print(f"[probe] {c}: yapi modu degil, atlaniyor")
            continue
        loaded.append((c, ck))
    if not loaded:
        return None
    keys = {_ds_key(ck["args"]) for _, ck in loaded}
    assert len(keys) == 1, "kontrol noktalari ayni veri kumesi ayarlarini paylasmali"
    args = loaded[0][1]["args"]

    cache = os.path.join(a.cache_dir or args.get("cache_dir") or a.data_dir,
                         f"emb_{args['esm']}_{args['esm_model'].split('/')[-1] if args['esm'] == 'esm' else 'oh'}.pt")
    emb = ProteinEmbedder(args["esm"], args["esm_model"], cache_path=cache)
    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=args["clean_level"])
    exts = load_external_sets(a.data_dir)

    # built once, reused for every checkpoint and every probe: the parsing is the expensive part
    dsets = {}
    for split in want:
        if split == "test":
            df, pp, lp = tabs["test"], args["protein_pattern"], args["ligand_pattern"]
        elif split in exts:
            df, pp, lp = exts[split]
        else:
            continue
        ds = StructAffinityDataset(df, emb, pp, lp, args["cutoff"], args["max_res"], args["max_atoms"],
                                   args["max_poses"], 0, 0.5, f"probe-{split}", eval_states=0,
                                   use_bfactor=args.get("use_bfactor", False), keep_coords=True,
                                   # the probe must feed the model exactly what it was trained on,
                                   # otherwise it measures a featurisation mismatch instead of geometry use
                                   pocket_mode=args.get("pocket_mode", "nearest"),
                                   n_true=args.get("n_true", 48), geom=args.get("geom", "real"))
        if len(ds):
            dsets[split] = (ds, DataLoader(ds, batch_size=args["bs"], shuffle=False, collate_fn=collate))

    res = {"config": {"ckpts": cks, "probes": probes, "splits": list(dsets)}}
    rows = []
    for cpath, ck in loaded:
        tag = os.path.basename(os.path.dirname(cpath))
        model = EnsAff(ck.get("in_dim", ck["esm_dim"]), "struct", args["d"], args["c"], args["blocks"],
                       args["heads"], args["slots"], n_recycle=args["recycle"],
                       dropout=args["dropout"]).to(dev)
        model.load_state_dict(ck["model"])
        base = {}
        for probe in probes:
            for split, (ds, dl) in dsets.items():
                ds.probe = probe
                _ids, y, mu, _sd = predict(model, dl, dev)
                m = regression_metrics(y, mu)
                if probe == "none":
                    base[split] = mu.copy()
                    m["dmu"], m["r_base"] = 0.0, 1.0
                else:
                    b = base[split]
                    m["dmu"] = float(np.abs(mu - b).mean())
                    m["r_base"] = float(stats.pearsonr(mu, b)[0]) if np.std(mu) > 0 else float("nan")
                res.setdefault(tag, {}).setdefault(probe, {})[split] = m
                rows.append((tag, probe, split, m))
                print(f"  {tag:<12} {probe:<14} {split:<10} n={m['n']:<5} rmse={m['rmse']:.3f} "
                      f"r={m['pearson']:.3f} d|mu|={m['dmu']:.4f} r_base={m['r_base']:.4f}", flush=True)
        for ds, _dl in dsets.values():
            ds.probe = "none"
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # mean +- std over checkpoints, written as text so the report can inline it verbatim
    lines = ["(d|mu| = bozulmamis tahmine gore ortalama |degisim|, pK;  r_base = bozulmus vs bozulmamis Pearson)",
             "(r_base ~ 1.000 ve d|mu| ~ 0  =>  model bu kanali kullanmiyor)", ""]
    for split in dsets:
        lines.append(f"  {split}  ({len(loaded)} kontrol noktasi, ortalama +- std)")
        lines.append(f"    {'sonda':<16} {'rmse':>14} {'pearson':>14} {'d|mu| (pK)':>16} {'r_base':>14}")
        for probe in probes:
            got = [m for t, pr, sp, m in rows if pr == probe and sp == split]
            if not got:
                continue
            f = lambda k: (np.mean([g[k] for g in got]), np.std([g[k] for g in got], ddof=1) if len(got) > 1 else 0.0)  # noqa: E731
            (rm, rs), (pm, ps), (dm, dsd), (bm, bs) = f("rmse"), f("pearson"), f("dmu"), f("r_base")
            lines.append(f"    {probe:<16} {rm:7.3f}+-{rs:<5.3f} {pm:7.3f}+-{ps:<5.3f} "
                         f"{dm:9.4f}+-{dsd:<5.4f} {bm:7.4f}+-{bs:<5.4f}")
        lines.append("")
    # verdict: the point of the probe is whether erasing geometry costs anything
    for split in dsets:
        g0 = [m for t, pr, sp, m in rows if pr == "none" and sp == split]
        gc = [m for t, pr, sp, m in rows if pr == "const:8" and sp == split]
        if not (g0 and gc):
            continue
        d = np.mean([m["rmse"] for m in gc]) - np.mean([m["rmse"] for m in g0])
        rb = np.mean([m["r_base"] for m in gc])
        verdict = ("GEOMETRIYI KULLANIYOR" if d > 0.05 and rb < 0.99 else
                   "sinirda" if d > 0.02 or rb < 0.995 else "GEOMETRIYI KULLANMIYOR")
        lines.append(f"  KARAR [{split}]: geometriyi silmenin bedeli {d:+.3f} pK, r_base={rb:.4f} -> {verdict}")
    lines.append("  (olcut: bedel > 0.05 pK VE r_base < 0.99 ise model geometriyi okuyor demektir)")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(a.out, "probe_report.txt"), "w") as fh:
        fh.write(txt)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
