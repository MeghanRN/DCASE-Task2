
#!/usr/bin/env python3
"""
beats_knn_asd.py
────────────────
Unsupervised anomalous-sound detection for DCASE Task-2 2025 using

    • BEATs (AudioSet-pretrained transformer) as feature extractor
    • Cosine-distance K-nearest-neighbour (k=1 by default) anomaly score
    • Optional lightweight fine-tuning on *normal* clips only
    • Numpy NPZ file stores the reference embedding matrix

Usage
-----
# build reference from normal training WAVs
python beats_knn_asd.py build-index --wav-dir path/to/train --index-file ref.npz

# (optional) fine-tune BEATs backbone on normal clips
python beats_knn_asd.py fine-tune  --wav-dir path/to/train --ckpt tuned.pt

# inference / scoring
python beats_knn_asd.py infer --wav-dir path/to/test --index-file ref.npz --csv scores.csv
"""

from __future__ import annotations
import argparse, os, pathlib, warnings, json
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoConfig
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ★───────────────────────────────────────────────────────────────────────★
#  Configuration
# ★───────────────────────────────────────────────────────────────────────★
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/beats-base"   # allowed; AudioSet-pre-trained
SAMPLE_RATE = 16000                  # BEATs default
EMB_LAYER = -1                       # last hidden layer (pooled)


# ★───────────────────────────────────────────────────────────────────────★
#  Feature extractor
# ★───────────────────────────────────────────────────────────────────────★
class BEATsEmbedder:
    def __init__(self, checkpoint: str | None = None):
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        self.model = AutoModelForAudioClassification.from_pretrained(
            MODEL_NAME, config=cfg
        )
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(f"✓ loaded fine-tuned weights from {checkpoint}")
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        """
        wav : 1-D float32 waveform at 16 kHz
        returns L2-normalised clip embedding (np.float32, shape [D])
        """
        if wav.ndim == 2:  # stereo → mono
            wav = wav.mean(1)
        wav_t = torch.from_numpy(wav).to(DEVICE).unsqueeze(0)
        out = self.model(wav_t, output_hidden_states=True)
        h = out.hidden_states[EMB_LAYER]          # [B, T, D]
        feat = h.mean(1).squeeze()                # temporal mean-pool
        feat = feat.cpu().float().numpy()
        return normalize(feat.reshape(1, -1)).squeeze()  # cosine-safe


# ★───────────────────────────────────────────────────────────────────────★
#  Data helpers
# ★───────────────────────────────────────────────────────────────────────★
def list_wavs(wav_dir: str) -> List[str]:
    exts = (".wav", ".flac", ".ogg")
    wavs = sorted(str(p) for p in pathlib.Path(wav_dir).rglob("*") if p.suffix in exts)
    if not wavs:
        raise FileNotFoundError(f"No audio files under {wav_dir}")
    return wavs


def load_resample(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, src_sr = sf.read(path, dtype="float32")
    if src_sr != sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav.T), src_sr, sr
        ).T.numpy()
    return wav


# ★───────────────────────────────────────────────────────────────────────★
#  Build reference index (normal embeddings)
# ★───────────────────────────────────────────────────────────────────────★
def build_index(wav_dir: str, index_file: str, ckpt: str | None, k: int):
    embedder = BEATsEmbedder(ckpt)
    feats = []
    for p in tqdm(list_wavs(wav_dir), desc="Embedding train WAVs"):
        wav = load_resample(p)
        feats.append(embedder.embed(wav))
    feats = np.stack(feats)  # [N, D]
    np.savez_compressed(index_file, feats=feats, k=k)
    print(f"✓ saved {feats.shape[0]} embeddings → {index_file}")


# ★───────────────────────────────────────────────────────────────────────★
#  Fine-tune backbone on normal data (self-reconstruction)
# ★───────────────────────────────────────────────────────────────────────★
def fine_tune(wav_dir: str, save_ckpt: str, epochs: int, lr: float):
    # freeze all but the last Transformer block for speed
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    for param in model.parameters():
        param.requires_grad = False
    for p in model.wav2vec.encoder.layers[-1].parameters():
        p.requires_grad = True

    model.to(DEVICE).train()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = torch.nn.MSELoss()

    wav_paths = list_wavs(wav_dir)
    for ep in range(epochs):
        losses = []
        for pth in tqdm(wav_paths, desc=f"FT epoch {ep+1}/{epochs}"):
            wav = load_resample(pth)
            if wav.ndim == 2:
                wav = wav.mean(1)
            x = torch.from_numpy(wav).to(DEVICE).unsqueeze(0)
            out = model(x, output_hidden_states=True)
            h = out.hidden_states[EMB_LAYER]           # [1, T, D]
            rec = out.logits.unsqueeze(1)              # use classifier head as naive recon
            loss = crit(rec, h.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f" epoch {ep+1}: recon loss {np.mean(losses):.4f}")

    torch.save(model.state_dict(), save_ckpt)
    print(f"✓ fine-tuned weights saved to {save_ckpt}")


# ★───────────────────────────────────────────────────────────────────────★
#  Inference / scoring
# ★───────────────────────────────────────────────────────────────────────★
def infer(wav_dir: str, index_file: str, csv_path: str):
    pack = np.load(index_file)
    feats_db = pack["feats"]             # [N, D]
    k = int(pack["k"])
    nn = NearestNeighbors(
        n_neighbors=k, metric="cosine", algorithm="auto"
    ).fit(feats_db)

    embedder = BEATsEmbedder()
    with open(csv_path, "w") as f:
        f.write("file,anomaly_score\n")
        for p in tqdm(list_wavs(wav_dir), desc="Scoring"):
            wav = load_resample(p)
            q = embedder.embed(wav).reshape(1, -1)
            dist, _ = nn.kneighbors(q, return_distance=True)
            score = float(dist.mean())  # average k distances
            f.write(f"{os.path.basename(p)},{score:.6f}\n")
    print(f"✓ wrote scores to {csv_path}")


# ★───────────────────────────────────────────────────────────────────────★
#  CLI
# ★───────────────────────────────────────────────────────────────────────★
def parse_args():
    P = argparse.ArgumentParser()
    sub = P.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build-index", help="Embed normal clips and save NPZ index")
    sb.add_argument("--wav-dir", required=True)
    sb.add_argument("--index-file", required=True)
    sb.add_argument("--ckpt", help="optional fine-tuned BEATs checkpoint")
    sb.add_argument("--k", type=int, default=1, help="nearest neighbours (default 1)")

    sf = sub.add_parser("fine-tune", help="Lightweight BEATs fine-tuning on normals")
    sf.add_argument("--wav-dir", required=True)
    sf.add_argument("--ckpt", required=True, help="output .pt")
    sf.add_argument("--epochs", type=int, default=3)
    sf.add_argument("--lr", type=float, default=1e-5)

    si = sub.add_parser("infer", help="Score test WAVs given reference index")
    si.add_argument("--wav-dir", required=True)
    si.add_argument("--index-file", required=True)
    si.add_argument("--csv", required=True)

    return P.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_grad_enabled(False if args.cmd != "fine-tune" else True)

    if args.cmd == "build-index":
        build_index(args.wav_dir, args.index_file, args.ckpt, args.k)
    elif args.cmd == "fine-tune":
        fine_tune(args.wav_dir, args.ckpt, args.epochs, args.lr)
    elif args.cmd == "infer":
        infer(args.wav_dir, args.index_file, args.csv)