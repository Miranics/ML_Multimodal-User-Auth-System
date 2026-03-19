# Submission Checklist

## Completed Pipeline Outputs
- [x] [data/processed/merged_features.csv](../data/processed/merged_features.csv)
- [x] [data/processed/image_features.csv](../data/processed/image_features.csv)
- [x] [data/processed/audio_features.csv](../data/processed/audio_features.csv)
- [x] [artifacts/face_model.joblib](../artifacts/face_model.joblib)
- [x] [artifacts/voice_model.joblib](../artifacts/voice_model.joblib)
- [x] [artifacts/product_model.joblib](../artifacts/product_model.joblib)
- [x] [reports/metrics.json](metrics.json)
- [x] [reports/sample_faces_grid.png](sample_faces_grid.png)

## Team Tasks Remaining
- [ ] Colab notebook written and shared
- [ ] Demo video recorded (authorized + unauthorized flow)
- [ ] Final report completed
- [ ] Team contribution section completed
- [ ] GitHub repo cleaned and final commit pushed

## Demo Commands
```bash
.venv/bin/python -m src.auth_system_cli --face-image "data/images/member_1/neutral.jpg" --voice-audio "data/audio/member_1/yes_approve.wav"
.venv/bin/python -m src.auth_system_cli --face-image "data/images/unauthorized/unauthorized.jpg" --voice-audio "data/audio/unauthorized/unauthorized voice.wav"
```

## Metrics Snapshot
From [reports/metrics.json](metrics.json):
- Face model: accuracy 0.8750, f1 0.8250, loss 0.4302
- Voice model: accuracy 0.8333, f1 0.7667, loss 0.3759
- Product model: accuracy 0.7955, f1 0.7878, loss 0.7998
