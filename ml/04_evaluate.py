"""Оценка модели: multiclass (boat, gunshot, seal, background) + anomaly fallback."""
import json
from pathlib import Path

with open(Path(__file__).parent / "models" / "metrics.json") as f:
    m = json.load(f)

LABELS = ["boat", "gunshot", "seal", "background"]

print("=" * 55)
print("CASPIAN SENTINEL — Аудио-классификация")
print("Метрики модели (4 класса + anomaly fallback)")
print("=" * 55)
print(f"Accuracy:       {m['accuracy']:.3f}")
print(f"Precision(macro): {m['precision_macro']:.3f}")
print(f"Recall(macro):    {m['recall_macro']:.3f}")
print(f"F1(macro):        {m['f1_macro']:.3f}")
print(f"CV F1:           {m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}")
print(f"Samples:         {m['n_samples']}")
print(f"Features:        {m['n_features']}")
print(f"Anomaly threshold: {m.get('anomaly_threshold', 0.6)}")
print()
cm = m['confusion_matrix']
print("Confusion matrix (rows=true, cols=pred):")
print("         " + " ".join(f"{l:>8}" for l in LABELS))
for i, row in enumerate(cm):
    print(f"{LABELS[i]:>8} " + " ".join(f"{v:>8}" for v in row))
print()
print("Диагональ = правильные предсказания")
print("Вне диагонали = ошибки")
print()
print("Anomaly fallback:")
print("  - confidence < 0.6 → anomaly")
print("  - top_label == 'background' → anomaly")