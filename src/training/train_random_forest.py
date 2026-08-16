from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models.random_forest import build_random_forest
from src.preprocessing.color_features import FEATURE_NAMES, extract_color_texture_features
from src.preprocessing.labels import moisture_band
from src.training.common import load_features, split, write_json


def main():
    parser = argparse.ArgumentParser(description="Train the 8-feature Random Forest regressor.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/random_forest")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest, X = load_features(args.data_dir, extract_color_texture_features)
    X_train, X_test, y_train, y_test = split(X, manifest.moisture_pct.to_numpy())
    model = build_random_forest(); model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true_class = np.array([moisture_band(v) for v in y_test])
    y_pred_class = np.array([moisture_band(v) for v in predictions])
    classes = ["dry", "optimum", "saturated"]
    metrics = {"regression": regression_metrics(y_test, predictions), "binned_classification": classification_metrics(y_true_class, y_pred_class), "feature_names": FEATURE_NAMES}
    write_json(output / "metrics.json", metrics)
    save_confusion_matrix(y_true_class, y_pred_class, classes, output / "confusion_matrix.png")
    joblib.dump(model, output / "random_forest.joblib")
    print(f"Saved model and evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
