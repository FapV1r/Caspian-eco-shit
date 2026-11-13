"""Инференс: 4-классовая классификация (boat, gunshot, seal, background) + anomaly fallback.

Логика anomaly:
- Модель предсказывает 4 класса: boat, gunshot, seal, background.
- Если top-1 confidence < ANOMALY_THRESHOLD → "anomaly".
- Если top_label == "background" → "anomaly" (background в контракте = anomaly).
- Иначе — top-1 класс.

Возвращает ClassificationResult в формате API-контракта.
"""
import io
import joblib
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "classifier.pkl"
SR = 22050
N_MFCC = 13

# Порог: ниже этого confidence → аномалия
ANOMALY_THRESHOLD = 0.6

# 4 класса модели (в порядке обучения в 02_preprocess.py)
MODEL_LABELS = ["boat", "gunshot", "seal", "background"]
# 4 метки контракта (anomaly вместо background)
CONTRACT_LABELS = ["seal", "boat", "gunshot", "anomaly"]

model = joblib.load(str(MODEL_PATH))


def underwater_filter(audio, sr):
    cutoff = min(4000, sr / 2 - 1)
    b, a = butter(4, cutoff, btype='low', fs=sr)
    return filtfilt(b, a, audio)


def _extract_features(audio, orig_sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
    audio = underwater_filter(audio, SR)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return mfcc.mean(axis=1).reshape(1, -1)


def _predict(features):
    proba = model.predict_proba(features)[0]
    top_idx = int(np.argmax(proba))
    top_conf = float(proba[top_idx])
    top_label = MODEL_LABELS[top_idx]

    # Anomaly fallback:
    # 1) Если top_conf < threshold → anomaly (модель не уверена)
    # 2) Если top_label == "background" → anomaly (background в контракте = anomaly)
    # 3) Иначе → top_label
    if top_conf < ANOMALY_THRESHOLD or top_label == "background":
        final_label = "anomaly"
        anomaly_conf = proba[3] + (1 - top_conf) * 0.5  # background prob + неопределённость
    else:
        final_label = top_label
        anomaly_conf = proba[3]  # background prob

    # Строим candidates в порядке контракта: seal, boat, gunshot, anomaly
    candidates = []
    for label in CONTRACT_LABELS:
        if label == "seal":
            conf = proba[2]
        elif label == "boat":
            conf = proba[0]
        elif label == "gunshot":
            conf = proba[1]
        elif label == "anomaly":
            conf = anomaly_conf
        candidates.append({"label": label, "confidence": round(float(conf), 2)})

    # Сортируем по убыванию confidence
    candidates.sort(key=lambda c: -c["confidence"])

    top = candidates[0]
    # Убираем нулевые варианты
    candidates = [c for c in candidates if c is top or c["confidence"] > 0]

    return {"label": final_label, "confidence": top_conf, "candidates": candidates}


def classify_audio(wav_path, return_candidates=True):
    """Классифицирует аудио-файл."""
    audio, orig_sr = sf.read(str(wav_path))
    features = _extract_features(audio, orig_sr)
    return _predict(features)


def classify_bytes(audio_bytes, sr=SR):
    """Классифицирует аудио из байтов (для API)."""
    audio, orig_sr = sf.read(io.BytesIO(audio_bytes))
    features = _extract_features(audio, orig_sr)
    return _predict(features)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(Path(__file__).parent / "data" / "boat" / "boat_000.wav")

    result = classify_audio(path)
    print(f"Файл: {path}")
    print(f"Результат: {result}")