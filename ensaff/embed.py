"""Per-residue protein embeddings with on-disk caching.

backend="esm"    : HuggingFace ESM-2 (facebook/esm2_t12_35M_UR50D by default). Use on Colab GPU.
backend="onehot" : 21-d amino-acid one-hot (no download; for tests and as an ablation).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
CHUNK = 1000  # ESM-2 handles <=1022 residues; longer chains are embedded in windows


def _key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def onehot_embed(seq: str) -> np.ndarray:
    out = np.zeros((len(seq), 21), dtype=np.float16)
    for i, c in enumerate(seq):
        j = AA.find(c)
        out[i, j if j >= 0 else 20] = 1
    return out


class ProteinEmbedder:
    def __init__(self, backend: str = "esm", model_name: str = "facebook/esm2_t12_35M_UR50D",
                 cache_path: str | None = None, device: str | None = None, batch_tokens: int = 8000,
                 save_every: int = 800):
        self.backend = backend
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_tokens = batch_tokens
        self.save_every = save_every
        self.cache: dict[str, np.ndarray] = {}
        self._persisted: set[str] = set()
        self.shard_dir = (cache_path + ".shards") if cache_path else None
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            if os.path.exists(cache_path):            # legacy single-file cache
                self.cache = torch.load(cache_path, weights_only=False)
                print(f"[embed] loaded {len(self.cache)} cached chains from {cache_path}")
            if os.path.isdir(self.shard_dir):         # sharded cache (the current format)
                import glob as _glob
                for sh in sorted(_glob.glob(os.path.join(self.shard_dir, "shard_*.pt"))):
                    try:
                        self.cache.update(torch.load(sh, weights_only=False))
                    except Exception as e:
                        print(f"[embed] bozuk parca atlandi {sh}: {e}")
                print(f"[embed] loaded {len(self.cache)} cached chains from {self.shard_dir}")
            self._persisted = set(self.cache)
        self._model = None
        self._tok = None

    @property
    def dim(self) -> int:
        if self.backend == "onehot":
            return 21
        if self.cache:
            return next(iter(self.cache.values())).shape[1]
        self._load()
        return self._model.config.hidden_size

    def _load(self):
        if self._model is None:
            from transformers import AutoTokenizer, EsmModel
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = EsmModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self.device).eval()
            if self.device == "cuda":
                self._model = self._model.half()

    @torch.no_grad()
    def _esm_batch(self, seqs: list[str]) -> list[np.ndarray]:
        self._load()
        enc = self._tok(seqs, return_tensors="pt", padding=True).to(self.device)
        h = self._model(**enc).last_hidden_state.float().cpu().numpy()
        return [h[i, 1:len(s) + 1].astype(np.float16) for i, s in enumerate(seqs)]

    def _embed_chain_uncached(self, seqs: list[str]):
        if self.backend == "onehot":
            for s in seqs:
                self.cache[_key(s)] = onehot_embed(s)
            self.save(quiet=False)
            return
        # split long chains into windows, embed by length-sorted batches
        pieces = []
        for s in seqs:
            for st in range(0, len(s), CHUNK):
                pieces.append((s, st, s[st:st + CHUNK]))
        pieces.sort(key=lambda t: len(t[2]))
        npieces = {s: (len(s) + CHUNK - 1) // CHUNK for s in seqs}
        out: dict[tuple, np.ndarray] = {}
        got: dict[str, int] = {s: 0 for s in seqs}
        since_save, i = 0, 0
        from tqdm.auto import tqdm
        bar = tqdm(total=len(pieces), desc="ESM", disable=len(pieces) < 50)
        while i < len(pieces):
            j = i
            while j < len(pieces) and (j - i + 1) * (len(pieces[j][2]) + 2) <= self.batch_tokens:
                j += 1
            j = max(j, i + 1)
            batch = pieces[i:j]
            embs = self._esm_batch([p[2] for p in batch])
            for (s, st, _), e in zip(batch, embs):
                out[(s, st)] = e
                got[s] += 1
                if got[s] == npieces[s]:  # chain complete -> move to cache, free windows
                    self.cache[_key(s)] = np.concatenate([out.pop((s, k)) for k in range(0, len(s), CHUNK)], 0)
                    since_save += 1
            bar.update(j - i)
            if since_save >= self.save_every:  # periodic checkpoint so Ctrl+C never loses much
                bar.clear()
                self.save(quiet=False)
                since_save = 0
                bar.refresh()
            i = j
        bar.close()
        self.save(quiet=False)

    def embed_chains(self, seqs: list[str]):
        todo = sorted({s for s in seqs if _key(s) not in self.cache})
        if todo:
            print(f"[embed] embedding {len(todo)} new chains with {self.backend}")
            self._embed_chain_uncached(todo)
            self.save()

    def get(self, seq: str) -> np.ndarray:
        return self.cache[_key(seq)]

    def save(self, quiet: bool = True):
        """Write only the chains not yet on disk, as a new shard.

        The cache can reach tens of GB in structure mode (full chains at 1280 dims), so rewriting the whole
        cache at every checkpoint would dominate the run. Each save costs only what is new.
        """
        if not self.cache_path:
            return
        new_keys = [k for k in self.cache if k not in self._persisted]
        if not new_keys:
            return
        os.makedirs(self.shard_dir, exist_ok=True)
        idx = len([f for f in os.listdir(self.shard_dir) if f.startswith("shard_")])
        path = os.path.join(self.shard_dir, f"shard_{idx:04d}.pt")
        mb = sum(self.cache[k].nbytes for k in new_keys) / 1e6
        if not quiet:
            print(f"[embed] saving {len(new_keys)} new chains ({mb:.0f} MB) -> {os.path.basename(path)} "
                  f"— DO NOT interrupt...", flush=True)
        tmp = path + ".tmp"
        torch.save({k: self.cache[k] for k in new_keys}, tmp)
        os.replace(tmp, path)             # atomic: an interrupted write never leaves a half shard
        self._persisted.update(new_keys)
        if not quiet:
            print(f"[embed] saved. toplam onbellek: {len(self.cache)} zincir", flush=True)
