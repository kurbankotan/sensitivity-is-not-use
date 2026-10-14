r"""Does aggregating over several bound states at INFERENCE change anything?

This is the direct ablation of the idea the project is named after. With a single crystal pose K=1 and the
aggregation mu = tau*logsumexp(mu_k/tau) - tau*log K reduces exactly to mu_1, so every result so far says
nothing about it: in the runs to date the extra states existed only during training, as augmentation.

Here a trained checkpoint is re-evaluated with K = 1, 3, 5, 9 states. Each extra state is a rigid-body
translation of the ligand with distances recomputed from the moved coordinates, seeded per complex from a
stable hash so the comparison repeats exactly. (An earlier version added noise directly to the distance
matrix, which no pose change can produce; its K-ablation therefore tested noise robustness, not state
aggregation.) If the aggregation carries real signal, accuracy should move with K.

    python -m ensaff.ablate_states --ckpt runs/struct_s0/best.pt --out runs/ablate_s0
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import StructAffinityDataset, collate, load_external_sets, load_lp_pdbbind
from .embed import ProteinEmbedder
from .evaluate import regression_metrics, uncertainty_metrics
from .model import EnsAff
from .train import predict


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/ablate_states")
    p.add_argument("--states", default="0,2,4,8", help="extra jittered states to add at evaluation")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--splits", default="test,BDB2020+")
    a = p.parse_args(argv)
    ck = torch.load(a.ckpt, weights_only=False)
    args = ck["args"]
    if args.get("mode") != "struct":
        print("[ablate] yalnizca yapi modu icin anlamli (dizi modunda K hep 1)")
        return None
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(a.cache_dir or args.get("cache_dir") or a.data_dir,
                         f"emb_{args['esm']}_{args['esm_model'].split('/')[-1] if args['esm'] == 'esm' else 'oh'}.pt")
    emb = ProteinEmbedder(args["esm"], args["esm_model"], cache_path=cache)
    model = EnsAff(ck.get("in_dim", ck["esm_dim"]), "struct", args["d"], args["c"], args["blocks"], args["heads"], args["slots"],
                   n_recycle=args["recycle"], dropout=args["dropout"]).to(dev)
    model.load_state_dict(ck["model"])

    tabs = load_lp_pdbbind(os.path.join(a.data_dir, "LP_PDBBind.csv"), clean_level=args["clean_level"])
    exts = load_external_sets(a.data_dir)
    want = [s.strip() for s in a.splits.split(",") if s.strip()]
    res = {"config": {"ckpt": a.ckpt, "states": a.states, "seed": args.get("seed")}}
    for n_extra in [int(x) for x in a.states.split(",")]:
        K = n_extra + 1
        res[f"K={K}"] = {}
        for split in want:
            if split == "test":
                df, pp, lp = tabs["test"], args["protein_pattern"], args["ligand_pattern"]
            elif split in exts:
                df, pp, lp = exts[split]
            else:
                continue
            ds = StructAffinityDataset(df, emb, pp, lp, args["cutoff"], args["max_res"], args["max_atoms"],
                                       args["max_poses"], 0, 0.5, f"{split}-K{K}", eval_states=n_extra,
                                       use_bfactor=args.get("use_bfactor", False))
            if not len(ds):
                continue
            dl = DataLoader(ds, batch_size=args["bs"], shuffle=False, collate_fn=collate)
            _ids, y, mu, sd = predict(model, dl, dev)
            res[f"K={K}"][split] = {**regression_metrics(y, mu), **uncertainty_metrics(y, mu, sd)}
            r = res[f"K={K}"][split]
            print(f"  K={K:<2} {split:<10} n={r['n']:<5} rmse={r['rmse']:.3f} r={r['pearson']:.3f} "
                  f"rho={r['spearman']:.3f} sd~err={r['spearman_sd_vs_err']:+.3f}", flush=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
