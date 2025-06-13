#!/usr/bin/env python3
"""
beats_knn_asd.py – **fully self‑contained**
──────────────────────────────────────────
Unsupervised anomalous‑sound detection for **DCASE Task 2 (2025)** using

* Microsoft **BEATs** (.pt checkpoints) for clip embeddings
* Cosine *k*‑nearest‑neighbour (default *k* = 1) anomaly score
* Reference embeddings saved as a compressed **NPZ**

The script now **bootstraps itself**: if it cannot import the `BEATs`
package it will automatically clone Microsoft’s UNILM repo, expose the
`BEATs/` sub‑package on `PYTHONPATH`, and continue. No manual pip
installation is required.

Example
-------
```bash
# ❶ one‑time download of a checkpoint (≈400 MB)
wget -O checkpoints/BEATs_iter3.pt https://aka.ms/beats/BEATs_iter3.pt

# ❷ build reference index from normal train WAVs
python beats_knn_asd.py build-index \
       --wav-dir   data/AutoTrash/train \
       --index-file work/indices/AutoTrash.npz \
       --ckpt      checkpoints/BEATs_iter3.pt

# ❸ score test set
python beats_knn_asd.py infer \
       --wav-dir   data/AutoTrash/test \
       --index-file work/indices/AutoTrash.npz \
       --ckpt      checkpoints/BEATs_iter3.pt \
       --csv       work/scores/AutoTrash.csv
```
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

###############################################################################
# Bootstrap BEATs (clone if missing)
###############################################################################

def _ensure_beats() -> None:
    """Import the BEATs package, cloning UNILM if necessary."""

    try:
        import BEATs  # noqa: F401 – check only
        return                     # already importable
    except ModuleNotFoundError:
        print("◆ BEATs not found – cloning UNILM repo …", file=sys.stderr)

    clone_root = pathlib.Path.home() / ".cache_beats_unilm"
    beats_path = clone_root / "unilm" / "BEATs"

    if not beats_path.exists():
        clone_root.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/microsoft/unilm.git", str(clone_root / "unilm")
        ])
    # expose to current process and all child processes
    sys.path.insert(0, str(beats_path.parent))
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{beats_path.parent}"

_ensure_beats()
from BEATs import BEATs, BEATsConfig  # noqa: E402  (import after bootstrap)

###############################################################################
# Config constants
###############################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16_000      # Hz – BEATs native rate
EMB_LAYER   = -1          # use last hidden layer of the transformer

###############################################################################
# Helper classes / functions
###############################################################################

class BEATsEmbedder:
    """Lightweight wrapper that turns a waveform into a single clip embedding."""

    def __init__(self, ckpt: str):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        cfg   = BEATsConfig(state["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(state["model"], strict=False)
        model.to(DEVICE).eval()
        self.model = model
        print(f"◆ Loaded BEATs checkpoint: {ckpt}")

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        # mono‑ise
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        feats, _ = self.model.extract_features(wav_t, padding_mask=None)
        vec = feats.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
        return normalize(vec.reshape(1, -1)).squeeze()

# ────────────────────────────────────────────
# audio I/O
# ────────────────────────────────────────────

def list_wavs(wav_dir: str) -> List[str]:
    exts = (".wav", ".flac", ".ogg")
    return sorted(str(p) for p in pathlib.Path(wav_dir).rglob("*") if p.suffix.lower() in exts)

def load_resample(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, orig_sr = sf.read(path, dtype="float32")
    if orig_sr != sr:
        wav = torchaudio.functional.resample(torch.from_numpy(wav.T), orig_sr, sr).T.numpy()
    return wav

###############################################################################
# Core tasks: build index  •  infer
###############################################################################

def build_index(wav_dir: str, index_file: str, ckpt: str, k: int):
    embedder = BEATsEmbedder(ckpt)
    feats: list[np.ndarray] = []
    for p in tqdm(list_wavs(wav_dir), desc="Embedding train WAVs"):
        feats.append(embedder.embed(load_resample(p)))
    feats_np = np.stack(feats)
    np.savez_compressed(index_file, feats=feats_np, k=k)
    print(f"◆ Saved {feats_np.shape[0]} embeddings → {index_file}")

def infer(wav_dir: str, index_file: str, ckpt: str, csv_path: str):
    db   = np.load(index_file)
    feats_db = db["feats"]
    k = int(db["k"])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(feats_db)
    embedder = BEATsEmbedder(ckpt)

    with open(csv_path, "w") as f:
        f.write("file,anomaly_score\n")
        for p in tqdm(list_wavs(wav_dir), desc="Scoring test WAVs"):
            q = embedder.embed(load_resample(p)).reshape(1, -1)
            dist, _ = nn.kneighbors(q, return_distance=True)
            f.write(f"{os.path.basename(p)},{float(dist.mean()):.6f}\n")
    print(f"◆ Scores written → {csv_path}")

###############################################################################
# CLI
###############################################################################

def parse_args():
    P = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = P.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build-index", help="Embed normal clips and save NPZ reference")
    sb.add_argument("--wav-dir", required=True)
    sb.add_argument("--index-file", required=True)
    sb.add_argument("--ckpt", required=True, help="Path to BEATs *.pt file")
    sb.add_argument("--k", type=int, default=1, help="nearest neighbours")

    si = sub.add_parser("infer", help="Score WAVs using a reference index")
    si.add_argument("--wav-dir", required=True)
    si.add_argument("--index-file", required=True)
    si.add_argument("--ckpt", required=True)
    si.add_argument("--csv", required=True)

    return P.parse_args()

###############################################################################
# Main entry
###############################################################################

if __name__ == "__main__":
    a = parse_args()
    if a.cmd == "build-index":
        build_index(a.wav_dir, a.index_file, a.ckpt, a.k)
    elif a.cmd == "infer":
        infer(a.wav_dir, a.index_file, a.ckpt, a.csv)
