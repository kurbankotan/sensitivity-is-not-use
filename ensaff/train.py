"""Train / evaluate EnsAff.

Sequence mode (no structures needed):
    python -m ensaff.train --mode seq --esm esm --out runs/seq_esm35m
Structure mode (PDBbind-style folders, e.g. after registering at PDBbind):
    python -m ensaff.train --mode struct --esm esm \
        --protein_pattern "/content/pdbbind/{id}/{id}_protein.pdb" \
        --ligand_pattern  "/content/pdbbind/{id}/{id}_ligand.sdf" --out runs/struct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (SeqAffinityDataset, StructAffinityDataset, collate, load_external_sets, load_lp_pdbbind)
from .embed import ProteinEmbedder
from .evaluate import conformal_quantile, regression_metrics, uncertainty_metrics
from .model import EnsAff, beta_nll, pose_accuracy, pose_hinge


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def to_device(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, dev, amp=True):
    model.eval()
    ids, ys, mus, sds, attn = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            out = model(batch)
        ids += batch["ids"]
        ys.append(batch["y"].cpu())
        mus.append(out["mu"].float().cpu())
        sds.append(out["var"].float().sqrt().cpu())
    return ids, torch.cat(ys).numpy(), torch.cat(mus).numpy(), torch.cat(sds).numpy()


def build_datasets(a, emb):
    """Returns (train, val, test, {name: external_dataset}).

    External sets probe two different kinds of generalisation: BDB2020+ holds unseen targets, while mpro and
    EGFR hold many ligands against a SINGLE target, which is the lead-optimisation question.
    """
    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=a.clean_level)
    exts = load_external_sets(a.data_dir)
    if a.limit:
        tabs = {k: v.sample(min(len(v), a.limit if k == "train" else max(50, a.limit // 4)), random_state=0)
                for k, v in tabs.items()}
    out = {}
    if a.mode == "seq":
        mk = lambda df, n: SeqAffinityDataset(df, emb, a.max_atoms, n)  # noqa: E731
        for name, (df, _pp, _lp) in exts.items():
            out[name] = mk(df, name)
    else:
        # a control arm has to damage train, val, test and the external sets identically, otherwise the
        # comparison measures a train/test mismatch instead of the contribution of the channel
        mk = lambda df, n: StructAffinityDataset(df, emb, a.protein_pattern, a.ligand_pattern, a.cutoff,  # noqa: E731
                                                 a.max_res, a.max_atoms, a.max_poses, a.jitter_states, 0.5, n,
                                                 eval_states=a.eval_states, use_bfactor=a.use_bfactor,
                                                 geom=a.geom, pocket_mode=a.pocket_mode,
                                                 decoy_spec=a.pose_decoy, n_true=a.n_true)
        for name, (df, pp, lp) in exts.items():
            out[name] = StructAffinityDataset(df, emb, pp, lp, a.cutoff, a.max_res, a.max_atoms,
                                              a.max_poses, 0, 0.5, name, eval_states=a.eval_states,
                                              use_bfactor=a.use_bfactor, geom=a.geom,
                                              pocket_mode=a.pocket_mode, decoy_spec=a.pose_decoy,
                                              n_true=a.n_true)
    return mk(tabs["train"], "train"), mk(tabs["val"], "val"), mk(tabs["test"], "test"), out



def _chance_level(smask, n_real):
    """Poz ayirt etme olcutunun null altindaki basari olasiligi, kompleks basina.

    Olcut "her kristal poz her celdiriciden yuksek puan alsin" seklindedir. K durum
    degisilebilir ise, n_real gercegin hepsinin ilk n_real sirayi kapma olasiligi
    n_real! * d! / K! = 1 / C(K, n_real) olur.

    Bu fonksiyon, daha once ekrana basilan 1/(1+jitter_states) ifadesinin yerine gecer.
    O ifade egitimdeki celdirici sayisini varsayiyordu; degerlendirme ise
    max(4, jitter_states) celdirici kullanir ve birden fazla kristal pozu olan
    kompleksleri hic hesaba katmiyordu. Ikisi de sans duzeyini OLDUGUNDAN YUKSEK
    gosteriyordu, yani sonucu oldugundan zayif gosteriyordu.
    """
    from math import comb
    K = smask.sum(1).detach().cpu().numpy()
    nr = n_real.detach().cpu().numpy()
    out = [1.0 / comb(int(k), int(r)) for k, r in zip(K, nr) if k > r >= 1]
    return float(np.mean(out)) if out else float("nan")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="seq", choices=["seq", "struct"])
    p.add_argument("--esm", default="esm", choices=["esm", "onehot"])
    p.add_argument("--esm_model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--cache_dir", default="", help="where to keep the ESM embedding cache (default: data_dir). "
                   "On Colab use a LOCAL path like /content/emb_cache — writing the multi-GB cache to Drive is slow.")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--clean_level", default="CL1")
    p.add_argument("--protein_pattern", default="pdbbind/{id}/{id}_protein.pdb")
    p.add_argument("--ligand_pattern", default="pdbbind/{id}/{id}_ligand.sdf")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--max_res", type=int, default=64)
    p.add_argument("--max_atoms", type=int, default=72)
    p.add_argument("--max_poses", type=int, default=4)
    p.add_argument("--jitter_states", type=int, default=0, help="struct mode: extra jittered states in training")
    p.add_argument("--posedisc", type=float, default=0.0,
                   help="weight of the pose-discrimination hinge: the crystal pose must outscore its "
                        "decoys. Without it nothing in the objective makes the states differ (v12: K=1..9 "
                        "identical to 3 d.p.). Needs --jitter_states > 0.")
    p.add_argument("--pose_margin", type=float, default=0.5, help="required pK gap, crystal over decoy")
    p.add_argument("--pose_decoy", default="rotate",
                   help="how training decoys are made: rotate (about the ligand centroid, so the coarse "
                        "distance statistics are unchanged) | translate | translate:s")
    p.add_argument("--n_true", type=int, default=48,
                   help="pocket_mode=wide: how many genuine contact residues are hidden among the decoys")
    p.add_argument("--geom", default="real",
                   help="KONTROL: mesafe kanalina kasitli hasar. real | const:8 (geometri yok) | "
                        "permute_res (dogru mesafeler, yanlis kalintilar) | permute_atom | scale:f")
    p.add_argument("--pocket_mode", default="nearest", choices=["nearest", "random", "wide"],
                   help="KONTROL: random = cep secimi bozulur, geometri gercek kalir; "
                        "wide = gercek cep sahte kalintilar arasina saklanir (kisayol kapanir)")
    p.add_argument("--use_bfactor", action="store_true",
                   help="D1: append observed per-residue flexibility (B-factor z-scores) to the embedding")
    p.add_argument("--eval_states", type=int, default=0,
                   help="struct mode: extra states at EVALUATION too. K=1 makes the aggregation an identity, "
                        "so this is what actually exercises it at inference (ablation of the central idea)")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--c", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--slots", type=int, default=48)
    p.add_argument("--recycle", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-4)   # 3e-4 overfit by epoch 1; lower peak LR trains longer
    p.add_argument("--wd", type=float, default=5e-2)     # stronger weight decay
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--mse_w", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr_patience", type=int, default=2, help="epochs without val improvement before LR is halved")
    p.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage (0.1 -> 90%% intervals)")
    p.add_argument("--limit", type=int, default=0, help="debug: subsample train to this many complexes")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(a.cache_dir or a.data_dir,
                         f"emb_{a.esm}_{a.esm_model.split('/')[-1] if a.esm == 'esm' else 'oh'}.pt")
    emb = ProteinEmbedder(a.esm, a.esm_model, cache_path=cache)
    tr, va, te, exts = build_datasets(a, emb)
    if a.mode == "struct":
        tr.train_mode = True

    kw = dict(collate_fn=collate, num_workers=a.workers, pin_memory=dev.type == "cuda")
    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True, **kw)
    dl = {n: DataLoader(ds, batch_size=a.bs, shuffle=False, **kw)
          for n, ds in ([("val", va), ("test", te)] + list(exts.items())) if len(ds)}
    print(f"[data] degerlendirme setleri: " + ", ".join(f"{n}({len(ds)})" for n, ds in dl.items()))

    in_dim = emb.dim + (2 if (a.use_bfactor and a.mode == 'struct') else 0)
    model = EnsAff(in_dim, a.mode, a.d, a.c, a.blocks, a.heads, a.slots, n_recycle=a.recycle,
                   dropout=a.dropout).to(dev)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    # Warmup on steps, then halve the LR whenever validation stops improving. A cosine schedule sized to
    # --epochs never anneals when early stopping fires first (the LR stayed near its peak and validation
    # oscillated), so decay is driven by validation instead of by a fixed horizon.
    warm = min(500, (a.epochs * max(1, len(dl_tr))) // 10)
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                         patience=a.lr_patience, min_lr=a.lr * 0.01)
    step_count = 0

    best, bad, hist = float("inf"), 0, []
    for ep in range(a.epochs):
        model.train()
        t0, losses, pose_acc = time.time(), [], []
        for batch in dl_tr:
            batch = to_device(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(batch)
            y = batch["y"]
            loss = beta_nll(out["mu"], out["var"], y, a.beta) + a.mse_w * ((out["mu"] - y) ** 2).mean()
            if a.posedisc > 0 and "state_mask" in batch:
                loss = loss + a.posedisc * pose_hinge(out["state_mu"].float(), batch["state_mask"],
                                                      batch["n_real"], a.pose_margin)
                pose_acc.append(pose_accuracy(out["state_mu"].float(), batch["state_mask"], batch["n_real"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_count += 1
            if step_count <= warm:
                warmup.step()
            losses.append(loss.item())
        _, yv, mv, sv = predict(model, dl["val"], dev)
        m = regression_metrics(yv, mv)
        cur_lr = opt.param_groups[0]["lr"]
        plateau.step(m["rmse"])
        hist.append({"epoch": ep, "loss": float(np.mean(losses)),
                     "pose_acc": float(np.nanmean(pose_acc)) if pose_acc else float("nan"),
                     **{f"val_{k}": v for k, v in m.items()},
                     "tau": model.tau.item(), "lr": cur_lr, "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {np.mean(losses):.3f} val rmse {m['rmse']:.3f} r {m['pearson']:.3f} "
              f"lr {cur_lr:.2e} ({time.time() - t0:.0f}s)")
        if m["rmse"] < best - 1e-4:
            best, bad = m["rmse"], 0
            torch.save({"model": model.state_dict(), "args": vars(a), "esm_dim": emb.dim, "in_dim": in_dim}, f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("[early stop]")
                break
    pd.DataFrame(hist).to_csv(f"{a.out}/history.csv", index=False)

    # ---- final evaluation with the best checkpoint + split-conformal calibration on val
    model.load_state_dict(torch.load(f"{a.out}/best.pt", weights_only=False)["model"])
    # Conformal calibration on a half of val that was NOT used for early stopping. Calibrating on the
    # selection set makes its residuals optimistically small, so q comes out too tight and the nominal
    # 90% intervals under-cover on test (seen directly: val coverage 0.901 vs test 0.872).
    ids_v, yv, mv, sv = predict(model, dl["val"], dev)
    rng = np.random.RandomState(a.seed)
    half = rng.rand(len(yv)) < 0.5
    if half.sum() < 50 or (~half).sum() < 50:
        half = np.ones(len(yv), dtype=bool)
        print("[conformal] val cok kucuk -> ayri kalibrasyon yarisi kullanilamadi")
    q = conformal_quantile(yv[half], mv[half], sv[half], a.alpha)
    pose_val = float("nan")
    if a.posedisc > 0 and a.mode == "struct":
        ds = dl["val"].dataset
        keep = ds.eval_states
        ds.eval_states = max(4, a.jitter_states)       # decoys only for this pass; metrics above used none
        accs, chances = [], []
        with torch.no_grad():
            for b in dl["val"]:
                b = to_device(b, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    o = model(b)
                accs.append(pose_accuracy(o["state_mu"].float(), b["state_mask"], b["n_real"]))
                chances.append(_chance_level(b["state_mask"], b["n_real"]))
        ds.eval_states = keep
        pose_val = float(np.nanmean(accs))
        chance = float(np.nanmean(chances))
        print(f"[poz] val ayirt etme dogrulugu = {pose_val:.3f} "
              f"(sans duzeyi {chance:.3f}, 1.0 = tam ayirt etme)")
        print(f"[poz] sans duzeyi degerlendirmede FIILEN kullanilan durum sayisindan hesaplanir; "
              f"ulasilan deger sansin {pose_val / max(chance, 1e-9):.2f} kati")
    best_epoch = int(pd.read_csv(f"{a.out}/history.csv")["val_rmse"].idxmin()) if hist else -1
    results = {"config": {"mode": a.mode, "esm": a.esm, "esm_model": a.esm_model if a.esm == "esm" else "onehot",
                          "clean_level": a.clean_level, "epochs_run": len(hist), "best_epoch": best_epoch,
                          "eval_states": a.eval_states, "lr": a.lr, "lr_patience": a.lr_patience, "wd": a.wd, "dropout": a.dropout, "recycle": a.recycle, "bs": a.bs,
                          "esm_dim": emb.dim, "in_dim": in_dim, "use_bfactor": bool(a.use_bfactor), "geom": a.geom, "pocket_mode": a.pocket_mode, "posedisc": a.posedisc,
                          "pose_decoy": a.pose_decoy, "jitter_states": a.jitter_states,
                          "max_res": a.max_res, "n_true": a.n_true, "bs_": a.bs, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                          "seed": a.seed}, "conformal_q": q, "conformal_n_cal": int(half.sum()),
               "conformal_note": "calibrated on a val half disjoint from early stopping",
               "pose_val_accuracy": pose_val,
               "bfactor_degenerate": ({k: int(getattr(v.dataset, "n_degenerate_bfactor", 0)) for k, v in dl.items()}
                                      if a.use_bfactor else {})}
    for name, loader in dl.items():
        ids, y, mu, sd = predict(model, loader, dev)
        results[name] = {**regression_metrics(y, mu), **uncertainty_metrics(y, mu, sd, q, a.alpha)}
        pd.DataFrame({"id": ids, "y": y, "pred": mu, "sd": sd, "lo": mu - q * sd, "hi": mu + q * sd}) \
            .to_csv(f"{a.out}/pred_{name}.csv", index=False)
    with open(f"{a.out}/metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
