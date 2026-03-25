# ML Multimodal User Auth System

End-to-end multimodal machine learning pipeline for secure product recommendation. The system authenticates a user using face and voice checks before allowing a product prediction.

## Overview                             

This project implements:
- Tabular data merge and feature engineering
- Face feature extraction and identity classification
- Voice feature extraction and identity classification
- Product recommendation model
- CLI flow for authorized and unauthorized simulation

## Architecture

Authentication and recommendation flow:

1. User submits face image
2. Face model predicts identity                    
3. If face is valid, product model prepares prediction
4. User submits voice sample
5. Voice model verifies identity
6. If face and voice match, product prediction is displayed

Security checkpoints:
- Face fail -> access denied
- Voice fail -> access denied

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── images/
│   ├── audio/
│   └── processed/
├── artifacts/
├── reports/
├── src/
├── requirements.txt
├── multimodal_pipeline.ipynb
└── README.md
```

## Input Data Requirements

### Tabular data

Place files in:
- [data/raw/customer_social_profiles.csv](data/raw/customer_social_profiles.csv)
- [data/raw/customer_transactions.csv](data/raw/customer_transactions.csv)

Notes:
- Both tables should be joinable by customer identity.
- Product target may be any of: `product`, `product_name`, `purchased_product`, `item`, `target`, `product_category`.

### Image data

Use member-based folders:

```text
data/images/
  member_1/
    neutral.jpg
    smile.jpg
    surprised.jpg
  member_2/
  member_3/
  unauthorized/
    unauthorized.jpg
```

### Audio data

Use member-based folders:

```text
data/audio/
  member_1/
    yes_approve.wav
    confirm_transaction.wav
  member_2/
  member_3/
  unauthorized/
    unauthorized voice.wav
```

## Setup

1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start (Step by Step)

1. Merge raw tabular datasets

```bash
python -m src.data_merge
```

2. Extract image features with augmentations

```bash
python -m src.image_pipeline
```

3. Extract audio features with augmentations

```bash
python -m src.audio_pipeline
```

4. Train all models

```bash
python -m src.train_models
```

5. Generate report visuals

```bash
python -m src.visualize_samples
```

6. Run authorized demo

```bash
python -m src.auth_system_cli --face-image "data/images/member_1/neutral.jpg" --voice-audio "data/audio/member_1/yes_approve.wav"
```

7. Run unauthorized demo

```bash
python -m src.auth_system_cli --face-image "data/images/unauthorized/unauthorized.jpg" --voice-audio "data/audio/unauthorized/unauthorized voice.wav"
```

## One-Command Pipeline

```bash
python -m src.run_pipeline
```

## Outputs

Core generated files:
- [data/processed/merged_features.csv](data/processed/merged_features.csv)
- [data/processed/image_features.csv](data/processed/image_features.csv)
- [data/processed/audio_features.csv](data/processed/audio_features.csv)
- [artifacts/face_model.joblib](artifacts/face_model.joblib)
- [artifacts/voice_model.joblib](artifacts/voice_model.joblib)
- [artifacts/product_model.joblib](artifacts/product_model.joblib)
- [reports/metrics.json](reports/metrics.json)

## Current Metrics (Latest Local Run)

From [reports/metrics.json](reports/metrics.json):
- Face model: accuracy 0.8750, F1-weighted 0.8250, loss 0.4302
- Voice model: accuracy 0.8333, F1-weighted 0.7667, loss 0.3759
- Product model: accuracy 0.7955, F1-weighted 0.7878, loss 0.7998

## Scripts Reference

- [src/data_merge.py](src/data_merge.py): data cleaning, merge, feature engineering
- [src/image_pipeline.py](src/image_pipeline.py): image augmentation and feature extraction
- [src/audio_pipeline.py](src/audio_pipeline.py): audio augmentation and feature extraction
- [src/train_models.py](src/train_models.py): training and evaluation for all models
- [src/auth_system_cli.py](src/auth_system_cli.py): end-to-end system simulation
- [src/visualize_samples.py](src/visualize_samples.py): image and audio visual reports

## Notebook and Reporting

- Main notebook: [multimodal_pipeline.ipynb](multimodal_pipeline.ipynb)

## Troubleshooting

1. Virtual environment not detected
- Run commands with `.venv/bin/python` directly.

2. Missing dependency errors
- Reinstall with `pip install -r requirements.txt` inside `.venv`.

3. Merge output empty
- Verify tabular ID alignment and formatting across both raw CSV files.

4. Colab notebook errors from old cached file
- Re-upload latest notebook file or upload with a new filename.
