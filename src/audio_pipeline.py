from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.config import AUDIO_DIR, AUDIO_FEATURES_PATH


def _augmentations(y: np.ndarray, sr: int) -> list[tuple[str, np.ndarray]]:
    y = y.astype(np.float32)
    noise = y + 0.005 * np.random.randn(len(y)).astype(np.float32)
    stretched = librosa.effects.time_stretch(y, rate=0.9)
    pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    return [("original", y), ("noise", noise), ("stretch", stretched), ("pitch+2", pitched)]


def _aggregate(v: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_std": float(np.std(v)),
        f"{prefix}_min": float(np.min(v)),
        f"{prefix}_max": float(np.max(v)),
    }


def extract_audio_features(y: np.ndarray, sr: int) -> dict[str, float]:
    feats: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        feats.update(_aggregate(mfcc[i], f"mfcc_{i+1}"))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feats.update(_aggregate(rolloff, "rolloff"))
    feats.update(_aggregate(centroid, "centroid"))
    feats.update(_aggregate(zcr, "zcr"))
    feats.update(_aggregate(rms, "rms"))
    feats["duration_sec"] = float(len(y) / sr)

    return feats


def process_audio_directory(audio_root: Path = AUDIO_DIR, out_path: Path = AUDIO_FEATURES_PATH) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    valid_ext = {".wav", ".mp3", ".flac", ".ogg"}

    audio_paths = [p for p in audio_root.rglob("*") if p.suffix.lower() in valid_ext]
    if not audio_paths:
        raise FileNotFoundError(
            f"No audio files found under {audio_root}. Put files in data/audio/<member_id>/..."
        )

    for path in audio_paths:
        member_id = path.parent.name
        y, sr = librosa.load(path, sr=None)

        for aug_name, aug_y in _augmentations(y, sr):
            f = extract_audio_features(aug_y, sr)
            row: dict[str, float | str] = {
                "member_id": member_id,
                "source_path": str(path),
                "augmentation": aug_name,
            }
            row.update(f)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features with augmentations.")
    parser.add_argument("--audio-root", type=Path, default=AUDIO_DIR)
    parser.add_argument("--out", type=Path, default=AUDIO_FEATURES_PATH)
    args = parser.parse_args()

    df = process_audio_directory(args.audio_root, args.out)
    print(f"Saved audio features: {args.out}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
