from src.audio_pipeline import process_audio_directory
from src.data_merge import load_and_merge
from src.image_pipeline import process_image_directory
from src.train_models import train_all


if __name__ == "__main__":
    merged = load_and_merge()
    merged.to_csv("data/processed/merged_features.csv", index=False)
    process_image_directory()
    process_audio_directory()
    metrics = train_all()
    print(metrics)
