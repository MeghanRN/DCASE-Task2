#!/usr/bin/env python3
"""
beats_knn_asd.py
────────────────
Unsupervised anomalous‑sound detection for DCASE Task‑2 2025 using

    • **BEATs** (AudioSet‑pretrained transformer, .pt checkpoints)
    • Cosine‑distance *k*‑nearest‑neighbour (default *k* = 1) anomaly score
    • Reference embedding matrix stored as a compressed NPZ file

The script purposefully avoids the Hugging Face Hub.  Simply download one of
Microsoft's raw checkpoints – e.g.  `BEATs_iter3.pt` – and point `--ckpt` to
its path.  No internet access is required after that.

Example
-------
# ❶ download model once (≈400 MB)
wget -O checkpoints/BEATs_iter3.pt \
     "https://aka.ms/beats/BEATs_iter3.pt"

# ❷ build reference index from normal training WAVs
python beats_knn_asd.py build-index \
       --wav-dir   data/AutoTrash/train \
       --index-file work/indices/AutoTrash.npz \
       --ckpt      checkpoints/BEATs_iter3.pt

# ❸ score test set
python beats_knn_asd.py infer \
       --wav-dir   data/AutoTrash/test \
       --index-file work/indices/AutoTrash.npz \
       --csv       work/scores/AutoTrash.csv
"""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

# BEATs imports (ensure that the beats/ folder is on PYTHONPATH)
from BEATs import BEATs, BEATsConfig  # type: ignore

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16_000          # BEATs native sampling rate (Hz)
EMB_LAYER = -1                # use the last hidden layer

def _info(msg: str) -> None:
    print(f"\u25C7 {msg}")

# ────────────────────────────────────────────────────────────────────────────
# BEATs wrapper
# ────────────────────────────────────────────────────────────────────────────
class BEATsEmbedder:
    """Lightweight helper around BEATs that outputs a single clip embedding."""

    def __init__(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(DEVICE).eval()

        self.model = model
        _info(f"Loaded BEATs model from {ckpt_path}")

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Return *L2‑normalised* clip embedding ( D ,)."""
        if wav.ndim == 2:  # stereo → mono
            wav = wav.mean(axis=1)

        # (B, T)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        feats, _ = self.model.extract_features(wav_t, padding_mask=None)  # (B, T, D)
        clip_vec = feats.mean(dim=1).squeeze().cpu().float().numpy()
        return normalize(clip_vec.reshape(1, -1)).squeeze()

# ────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ────────────────────────────────────────────────────────────────────────────

def list_wavs(wav_dir: str) -> List[str]:
    exts = (".wav", ".flac", ".ogg")
    wavs = sorted(str(p) for p in pathlib.Path(wav_dir).rglob("*") if p.suffix.lower() in exts)
    if not wavs:
        raise FileNotFoundError(f"No audio files found under '{wav_dir}'")
    return wavs

def load_resample(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, src_sr = sf.read(path, dtype="float32")
    if src_sr != sr:
        wav = torchaudio.functional.resample(torch.from_numpy(wav.T), src_sr, sr).T.numpy()
    return wav

# ────────────────────────────────────────────────────────────────────────────
# Index building
# ────────────────────────────────────────────────────────────────────────────

def build_index(wav_dir: str, index_file: str, ckpt: str, k: int):
    embedder = BEATsEmbedder(ckpt)
    feats: list[np.ndarray] = []

    for p in tqdm(list_wavs(wav_dir), desc="Embedding training WAVs"):
        wav = load_resample(p)
        feats.append(embedder.embed(wav))

    feats_np = np.stack(feats).astype("float32")  # (N, D)
    np.savez_compressed(index_file, feats=feats_np, k=k)
    _info(f"Saved {feats_np.shape[0]} embeddings → {index_file}")

# ────────────────────────────────────────────────────────────────────────────
# Inference / scoring
# ────────────────────────────────────────────────────────────────────────────

def infer(wav_dir: str, index_file: str, ckpt: str, csv_path: str):
    pack = np.load(index_file)
    feats_db = pack["feats"]  # (N, D)
    k = int(pack["k"])

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto").fit(feats_db)
    embedder = BEATsEmbedder(ckpt)

    with open(csv_path, "w") as f:
        f.write("file,anomaly_score\n")
        for p in tqdm(list_wavs(wav_dir), desc="Scoring test WAVs"):
            wav = load_resample(p)
            q = embedder.embed(wav).reshape(1, -1)
            dist, _ = nn.kneighbors(q, return_distance=True)
            score = float(dist.mean())
            f.write(f"{os.path.basename(p)},{score:.6f}\n")
    _info(f"Wrote scores → {csv_path}")

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    P = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = P.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build-index", help="Embed normal clips and save NPZ index")
    sb.add_argument("--wav-dir", required=True, help="Directory with normal training WAVs")
    sb.add_argument("--index-file", required=True, help="Output .npz file")
    sb.add_argument("--ckpt", required=True, help="Path to BEATs *.pt checkpoint")
    sb.add_argument("--k", type=int, default=1, help="Number of nearest neighbours")

    si = sub.add_parser("infer", help="Score WAVs given reference index")
    si.add_argument("--wav-dir", required=True, help="Directory with WAVs to score")
    si.add_argument("--index-file", required=True, help="Reference .npz built by build-index")
    si.add_argument("--ckpt", required=True, help="Same BEATs checkpoint used for the index")
    si.add_argument("--csv", required=True, help="Output CSV with anomaly scores")

    return P.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "build-index":
        build_index(args.wav_dir, args.index_file, args.ckpt, args.k)
    elif args.cmd == "infer":
        infer(args.wav_dir, args.index_file, args.ckpt, args.csv)
