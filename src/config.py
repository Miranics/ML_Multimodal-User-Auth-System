from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

SOCIAL_PROFILES_PATH = RAW_DIR / "customer_social_profiles.csv"
TRANSACTIONS_PATH = RAW_DIR / "customer_transactions.csv"
IMAGE_FEATURES_PATH = PROCESSED_DIR / "image_features.csv"
AUDIO_FEATURES_PATH = PROCESSED_DIR / "audio_features.csv"
MERGED_FEATURES_PATH = PROCESSED_DIR / "merged_features.csv"

FACE_MODEL_PATH = ARTIFACTS_DIR / "face_model.joblib"
VOICE_MODEL_PATH = ARTIFACTS_DIR / "voice_model.joblib"
PRODUCT_MODEL_PATH = ARTIFACTS_DIR / "product_model.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "product_label_encoder.joblib"
MODEL_REPORT_PATH = REPORTS_DIR / "metrics.json"

for p in [PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR, IMAGES_DIR, AUDIO_DIR, RAW_DIR]:
    p.mkdir(parents=True, exist_ok=True)
