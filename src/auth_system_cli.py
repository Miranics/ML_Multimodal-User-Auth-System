from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

from src.audio_pipeline import extract_audio_features
from src.config import (
    AUDIO_FEATURES_PATH,
    FACE_MODEL_PATH,
    IMAGE_FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MERGED_FEATURES_PATH,
    PRODUCT_MODEL_PATH,
    VOICE_MODEL_PATH,
)
from src.image_pipeline import extract_features as extract_image_features
from src.train_models import _build_product_frame, _infer_product_target


def _predict_identity(model, x: np.ndarray) -> tuple[str, float]:
    x2 = x.reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        x_input = pd.DataFrame(x2, columns=model.feature_names_in_)
    else:
        x_input = x2

    pred = str(model.predict(x_input)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_input)
        conf = float(np.max(proba))
    else:
        conf = 1.0
    return pred, conf


def run_transaction(face_image: Path, voice_audio: Path, threshold: float = 0.55) -> None:
    if not all(p.exists() for p in [FACE_MODEL_PATH, VOICE_MODEL_PATH, PRODUCT_MODEL_PATH, LABEL_ENCODER_PATH]):
        raise FileNotFoundError("Models missing. Train models first using src/train_models.py")

    face_model = joblib.load(FACE_MODEL_PATH)
    voice_model = joblib.load(VOICE_MODEL_PATH)
    product_model = joblib.load(PRODUCT_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    import cv2

    img = cv2.imread(str(face_image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {face_image}")
    img_feats = extract_image_features(img)

    face_user, face_conf = _predict_identity(face_model, img_feats)
    print(f"Face identity: {face_user} (confidence={face_conf:.3f})")
    if face_user.lower() == "unauthorized" or face_conf < threshold:
        print("ACCESS DENIED at face verification step.")
        return

    merged_df = pd.read_csv(MERGED_FEATURES_PATH)
    img_df = pd.read_csv(IMAGE_FEATURES_PATH)
    aud_df = pd.read_csv(AUDIO_FEATURES_PATH)
    product_df = _build_product_frame(merged_df, img_df, aud_df)
    target_col = _infer_product_target(product_df)

    user_rows = product_df[product_df["member_id"].astype(str) == str(face_user)]
    if user_rows.empty:
        user_rows = product_df.head(1)

    X_user = user_rows.drop(columns=[target_col]).head(1)
    pred_idx = product_model.predict(X_user)[0]
    pred_product = label_encoder.inverse_transform([pred_idx])[0]

    print(f"Preliminary product prediction ready: {pred_product}")

    y, sr = librosa.load(voice_audio, sr=None)
    aud_feats_dict = extract_audio_features(y, sr)
    aud_feats = np.array(list(aud_feats_dict.values()), dtype=np.float32)

    voice_user, voice_conf = _predict_identity(voice_model, aud_feats)
    print(f"Voice identity: {voice_user} (confidence={voice_conf:.3f})")

    if voice_user == face_user and voice_user.lower() != "unauthorized" and voice_conf >= threshold:
        print("VOICE VERIFIED ✅")
        print(f"AUTHORIZED USER: {face_user}")
        print(f"DISPLAY PREDICTED PRODUCT: {pred_product}")
    else:
        print("ACCESS DENIED at voice verification step.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI simulation for multimodal user auth flow.")
    parser.add_argument("--face-image", type=Path, required=True)
    parser.add_argument("--voice-audio", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    run_transaction(args.face_image, args.voice_audio, args.threshold)


if __name__ == "__main__":
    main()
