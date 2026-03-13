from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

from src.config import IMAGE_FEATURES_PATH, IMAGES_DIR


def _augmentations(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rot = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), 15, 1.0),
        (image.shape[1], image.shape[0]),
    )
    flip = cv2.flip(image, 1)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return [("original", image), ("rotate15", rot), ("flip", flip), ("grayscale", gray_bgr)]


def _hist_features(image: np.ndarray) -> np.ndarray:
    chans = cv2.split(image)
    feat_parts = []
    for chan in chans:
        h = cv2.calcHist([chan], [0], None, [32], [0, 256]).flatten()
        h = h / (h.sum() + 1e-8)
        feat_parts.append(h)
    return np.concatenate(feat_parts)


def _hog_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    h = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return h.astype(np.float32)


def extract_features(image: np.ndarray) -> np.ndarray:
    return np.concatenate([_hist_features(image), _hog_features(image)])


def process_image_directory(images_root: Path = IMAGES_DIR, out_path: Path = IMAGE_FEATURES_PATH) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    valid_ext = {".jpg", ".jpeg", ".png"}

    image_paths = [p for p in images_root.rglob("*") if p.suffix.lower() in valid_ext]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {images_root}. Put files in data/images/<member_id>/..."
        )

    for path in image_paths:
        member_id = path.parent.name
        image = cv2.imread(str(path))
        if image is None:
            continue

        for aug_name, aug_img in _augmentations(image):
            feats = extract_features(aug_img)
            row: dict[str, float | str] = {
                "member_id": member_id,
                "source_path": str(path),
                "augmentation": aug_name,
            }
            for i, v in enumerate(feats):
                row[f"img_f_{i}"] = float(v)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image features with augmentations.")
    parser.add_argument("--images-root", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=IMAGE_FEATURES_PATH)
    args = parser.parse_args()

    df = process_image_directory(args.images_root, args.out)
    print(f"Saved image features: {args.out}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
