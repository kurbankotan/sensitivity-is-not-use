"""Self-contained STRUCTURE-MODE demo on the auto-downloading BDB2020+ structures.

This exists so a bare "Run all" can exercise the full structure + multi-state pipeline end to end
without PDBbind registration. BDB2020+ has only 115 complexes and the split here is a plain random
split (NOT leak-proof), so this is a PROOF-OF-RUN, not a benchmark. The real structure-mode result
needs PDBbind training structures (see the notebook's structure-mode section).

    python -m ensaff.demo_struct --esm esm --esm_model facebook/esm2_t33_650M_UR50D --cache_dir /content/emb_cache
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StructAffinityDataset, collate, load_bdb2020
from .embed import ProteinEmbedder
from .evaluate import conformal_quantile, regression_metrics, uncertainty_metrics
from .model import EnsAff, beta_nll
from .train import predict, set_seed, to_device


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--esm", default="esm", choices=["esm", "onehot"])
    p.add_argument("--esm_model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--out", default="runs/struct_demo")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_poses", type=int, default=4)
    p.add_argument("--jitter_states", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[demo_struct] PROOF-OF-RUN on BDB2020+ (115 complexes, random split) — NOT a leak-proof benchmark.")

    df = load_bdb2020(os.path.join(a.data_dir, "BDB2020+")).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    n = len(df)
    tr_df, va_df, te_df = df[: int(0.8 * n)], df[int(0.8 * n): int(0.9 * n)], df[int(0.9 * n):]
    cache = os.path.join(a.cache_dir or a.data_dir,
                         f"emb_{a.esm}_{a.esm_model.split('/')[-1] if a.esm == 'esm' else 'oh'}.pt")
    emb = ProteinEmbedder(a.esm, a.esm_model, cache_path=cache)
    root = os.path.join(a.data_dir, "BDB2020+", "dataset", "{id}")
    mk = lambda d, j, nm: StructAffinityDataset(d, emb, os.path.join(root, "protein.pdb"),  # noqa: E731
                                                os.path.join(root, "ligand.sdf"), jitter_states=j, name=nm)
    tr, va, te = mk(tr_df, a.jitter_states, "train"), mk(va_df, 0, "val"), mk(te_df, 0, "test")
    tr.train_mode = True
    kw = dict(collate_fn=collate, num_workers=2, pin_memory=dev.type == "cuda")
    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=len(tr) > a.bs, **kw)
    dl_va, dl_te = DataLoader(va, batch_size=a.bs, collate_fn=collate), DataLoader(te, batch_size=a.bs, collate_fn=collate)

    model = EnsAff(emb.dim, "struct", d=128, c=64, n_blocks=4, n_heads=4).to(dev)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        model.train()
        for batch in dl_tr:
            batch = to_device(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(batch)
            loss = beta_nll(out["mu"], out["var"], batch["y"]) + 0.5 * ((out["mu"] - batch["y"]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if len(va):
            _, yv, mv, _ = predict(model, dl_va, dev)
            r = regression_metrics(yv, mv)["rmse"]
            if r < best:
                best, best_state = r, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    _, yv, mv, sv = predict(model, dl_va, dev)
    q = conformal_quantile(yv, mv, sv, a.alpha) if len(va) > 5 else float("nan")
    ids, y, mu, sd = predict(model, dl_te, dev)
    res = {"demo": True, "note": "BDB2020+ random split, proof-of-run only",
           "config": {"mode": "struct", "esm": a.esm, "esm_model": a.esm_model if a.esm == "esm" else "onehot",
                      "n_train": int(len(tr_df)), "n_test": int(len(te_df)), "epochs": a.epochs},
           "test": {**regression_metrics(y, mu), **uncertainty_metrics(y, mu, sd, q if not math.isnan(q) else None, a.alpha)}}
    with open(f"{a.out}/metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["test"], indent=2))
    return res


if __name__ == "__main__":
    main()
