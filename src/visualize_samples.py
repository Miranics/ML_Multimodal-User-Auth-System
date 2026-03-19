from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import AUDIO_DIR, IMAGES_DIR, REPORTS_DIR


def plot_image_grid(images_root: Path, out_path: Path) -> None:
    import cv2

    image_paths = [p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_paths = image_paths[:9]
    if not image_paths:
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, p in zip(axes.flatten(), image_paths):
        img = cv2.imread(str(p))
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(p.parent.name)
        ax.axis("off")

    for ax in axes.flatten()[len(image_paths) :]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_audio_wave_and_spec(audio_root: Path, out_dir: Path) -> None:
    audio_paths = [p for p in audio_root.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    for p in audio_paths[:6]:
        y, sr = librosa.load(p, sr=None)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title(f"Waveform: {p.name} ({p.parent.name})")

        S = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log", ax=axes[1])
        axes[1].set_title("Spectrogram")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{p.stem}_wave_spec.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample image/audio visualizations.")
    parser.add_argument("--images-root", type=Path, default=IMAGES_DIR)
    parser.add_argument("--audio-root", type=Path, default=AUDIO_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    plot_image_grid(args.images_root, args.report_dir / "sample_faces_grid.png")
    plot_audio_wave_and_spec(args.audio_root, args.report_dir)
    print(f"Saved visualizations under: {args.report_dir}")


if __name__ == "__main__":
    main()
