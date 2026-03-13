# ML_Multimodal-User-Auth-System

Multimodal authentication + product recommendation pipeline for the Formative 2 assignment.

This repository implements:
- Tabular merge + feature engineering
- Face feature extraction + face identity model
- Voice feature extraction + voice identity/verification model
- Product recommendation model
- Command-line flow that simulates:
	- Face check -> product prediction step unlocked
	- Voice check -> prediction approved or denied

## 1) Project Structure

LICENSE
README.md
requirements.txt
src/
data/
	raw/
	images/
	audio/
	processed/
artifacts/
reports/

## 2) Expected Input Data

### Tabular files (required)
Place both files in [data/raw/customer_social_profiles.csv](data/raw/customer_social_profiles.csv) and [data/raw/customer_transactions.csv](data/raw/customer_transactions.csv).

- A shared key is required (preferred: `customer_id`)
- Product target column should be one of:
	- `product`
	- `product_name`
	- `purchased_product`
	- `item`
	- `target`

### Image files (required)
Store face images using member folders:

data/images/
	member_1/
		neutral.jpg
		smile.jpg
		surprised.jpg
	member_2/
		...
	unauthorized/
		intruder1.jpg

### Audio files (required)
Store audio samples using member folders:

data/audio/
	member_1/
		yes_approve.wav
		confirm_transaction.wav
	member_2/
		...
	unauthorized/
		fake_attempt.wav

## 3) Setup

1. Create and activate a Python environment
2. Install dependencies:

pip install -r requirements.txt

## 4) Run Steps

### Step A: Merge tabular data

python -m src.data_merge

Output: [data/processed/merged_features.csv](data/processed/merged_features.csv)

### Step B: Extract image features with augmentations

python -m src.image_pipeline

Output: [data/processed/image_features.csv](data/processed/image_features.csv)

### Step C: Extract audio features with augmentations

python -m src.audio_pipeline

Output: [data/processed/audio_features.csv](data/processed/audio_features.csv)

### Step D: Train all models

python -m src.train_models

Outputs:
- [artifacts/face_model.joblib](artifacts/face_model.joblib)
- [artifacts/voice_model.joblib](artifacts/voice_model.joblib)
- [artifacts/product_model.joblib](artifacts/product_model.joblib)
- [reports/metrics.json](reports/metrics.json)

### Optional one-shot pipeline

python -m src.run_pipeline

## 5) Visualize sample images/audio for report

python -m src.visualize_samples

Outputs are saved in [reports](reports).

## 6) Run CLI system demo

python -m src.auth_system_cli --face-image "data/images/member_1/neutral.jpg" --voice-audio "data/audio/member_1/yes_approve.wav"

Expected logic:
1. Face identity predicted
2. If unauthorized or low confidence -> denied
3. Product prediction prepared
4. Voice identity predicted
5. If voice matches face and confidence is enough -> approved and product displayed

## 7) Assignment Deliverables Mapping

- Merged dataset + feature engineering: [data/processed/merged_features.csv](data/processed/merged_features.csv)
- Image features: [data/processed/image_features.csv](data/processed/image_features.csv)
- Audio features: [data/processed/audio_features.csv](data/processed/audio_features.csv)
- Scripts for three models and CLI:
	- [src/image_pipeline.py](src/image_pipeline.py)
	- [src/audio_pipeline.py](src/audio_pipeline.py)
	- [src/train_models.py](src/train_models.py)
	- [src/auth_system_cli.py](src/auth_system_cli.py)

## Notes

- The notebook can be done in Google Colab as requested.
- This codebase is prepared for your local/script deliverables.
- Add real team data to improve model quality and final metrics.