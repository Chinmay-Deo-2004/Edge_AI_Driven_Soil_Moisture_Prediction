from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.cross_validation import macro_f1_cross_validation
from src.evaluation.metrics import classification_metrics
from src.models.svm import build_svm
from src.preprocessing.rgb_histogram import extract_rgb_histogram
from src.training.common import load_features, split, write_json


def main():
    parser = argparse.ArgumentParser(description="Train the RGB-histogram RBF SVM.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/svm")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest, X = load_features(args.data_dir, extract_rgb_histogram, bins=args.bins)
    y = manifest.moisture_class.to_numpy()
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_svm(pca_components=args.pca_components)
    model.fit(X_train, y_train); predictions = model.predict(X_test)
    metrics = {"test": classification_metrics(y_test, predictions), "histogram_bins": args.bins, "pca_components": args.pca_components}
    if args.cross_validate:
        metrics["five_fold_macro_f1"] = macro_f1_cross_validation(build_svm(pca_components=args.pca_components), X, y)
    write_json(output / "metrics.json", metrics)
    save_confusion_matrix(y_test, predictions, ["dry", "optimum", "saturated"], output / "confusion_matrix.png")
    joblib.dump(model, output / "svm_pipeline.joblib")
    print(f"Saved model and evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
