from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.config import (
    AUDIO_FEATURES_PATH,
    FACE_MODEL_PATH,
    IMAGE_FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MERGED_FEATURES_PATH,
    MODEL_REPORT_PATH,
    PRODUCT_MODEL_PATH,
    VOICE_MODEL_PATH,
)
from src.utils import save_json


TARGET_CANDIDATES = [
    "product",
    "product_name",
    "purchased_product",
    "item",
    "target",
    "product_category",
]


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if y_proba is not None:
        try:
            metrics["loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            metrics["loss"] = float("nan")
    return metrics


def _train_identity_model(df: pd.DataFrame, model_path: Path, model_name: str) -> dict[str, float]:
    if "member_id" not in df.columns:
        raise ValueError(f"{model_name}: expected column 'member_id' in features.")

    X = df.drop(columns=[c for c in ["member_id", "source_path", "augmentation"] if c in df.columns])
    y = df["member_id"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    metrics = _safe_metrics(y_test.to_numpy(), y_pred, y_proba)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def _infer_product_target(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in TARGET_CANDIDATES:
        if c in lower_map:
            return lower_map[c]
    raise ValueError(
        "Could not infer product target column. Use one of: "
        + ", ".join(TARGET_CANDIDATES)
    )


def _build_product_frame(merged_df: pd.DataFrame, image_df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate multimodal features by member_id
    img_num = image_df.select_dtypes(include=[np.number]).copy()
    img_num["member_id"] = image_df["member_id"].astype(str)
    img_agg = img_num.groupby("member_id", as_index=False).mean().add_prefix("img_")
    img_agg = img_agg.rename(columns={"img_member_id": "member_id"})

    aud_num = audio_df.select_dtypes(include=[np.number]).copy()
    aud_num["member_id"] = audio_df["member_id"].astype(str)
    aud_agg = aud_num.groupby("member_id", as_index=False).mean().add_prefix("aud_")
    aud_agg = aud_agg.rename(columns={"aud_member_id": "member_id"})

    product_df = merged_df.copy()

    # Best-effort ID mapping
    if "member_id" not in product_df.columns:
        id_candidates = [c for c in product_df.columns if c.lower() in ["customer_id", "user_id", "id"]]
        if id_candidates:
            product_df["member_id"] = product_df[id_candidates[0]].astype(str)
        else:
            product_df["member_id"] = "unknown"

    product_df["member_id"] = product_df["member_id"].astype(str)
    product_df = product_df.merge(img_agg, on="member_id", how="left")
    product_df = product_df.merge(aud_agg, on="member_id", how="left")
    return product_df


def train_all(
    merged_path: Path = MERGED_FEATURES_PATH,
    image_path: Path = IMAGE_FEATURES_PATH,
    audio_path: Path = AUDIO_FEATURES_PATH,
) -> dict[str, dict[str, float]]:
    image_df = pd.read_csv(image_path)
    audio_df = pd.read_csv(audio_path)
    merged_df = pd.read_csv(merged_path)

    metrics: dict[str, dict[str, float]] = {}
    metrics["facial_recognition"] = _train_identity_model(image_df, FACE_MODEL_PATH, "FaceModel")
    metrics["voice_verification"] = _train_identity_model(audio_df, VOICE_MODEL_PATH, "VoiceModel")

    product_df = _build_product_frame(merged_df, image_df, audio_df)
    target_col = _infer_product_target(product_df)

    y_raw = product_df[target_col].astype(str)
    X = product_df.drop(columns=[target_col])
    # If multimodal keys do not overlap merged IDs, joined feature columns can be all-NaN.
    X = X.dropna(axis=1, how="all")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    product_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400, random_state=42, class_weight="balanced_subsample"
                ),
            ),
        ]
    )
    product_model.fit(X_train, y_train)

    y_pred = product_model.predict(X_test)
    y_proba = product_model.predict_proba(X_test)
    product_metrics = _safe_metrics(y_test, y_pred, y_proba)

    PRODUCT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(product_model, PRODUCT_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    metrics["product_recommendation"] = product_metrics
    save_json(MODEL_REPORT_PATH, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train face, voice, and product models.")
    parser.add_argument("--merged", type=Path, default=MERGED_FEATURES_PATH)
    parser.add_argument("--image-features", type=Path, default=IMAGE_FEATURES_PATH)
    parser.add_argument("--audio-features", type=Path, default=AUDIO_FEATURES_PATH)
    args = parser.parse_args()

    metrics = train_all(args.merged, args.image_features, args.audio_features)
    print("Training complete. Metrics:")
    for section, vals in metrics.items():
        print(f"\n[{section}]")
        for k, v in vals.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
