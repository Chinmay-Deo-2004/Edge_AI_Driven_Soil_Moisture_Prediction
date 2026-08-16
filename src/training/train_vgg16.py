from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.models.vgg16 import build_vgg16_classifier
from src.preprocessing.labels import build_manifest
from src.training.common import write_json


def main():
    parser = argparse.ArgumentParser(description="Train the frozen VGG16 soil-moisture baseline.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/vgg16")
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required: pip install tensorflow") from exc
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_dir)
    classes = ["dry", "optimum", "saturated"]; class_to_id = {label: idx for idx, label in enumerate(classes)}
    paths = manifest.path.to_numpy(); labels = manifest.moisture_class.map(class_to_id).to_numpy()
    train_paths, test_paths, train_y, test_y = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, (128, 128)) / 255.0
        return image, label
    augment = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.055), tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomZoom(0.2), tf.keras.layers.RandomTranslation(0.2, 0.2)])
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).map(load, num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: (augment(x, training=True), y)).batch(16).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_y)).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(16)
    model = build_vgg16_classifier()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(patience=3)]
    history = model.fit(train_ds, validation_data=test_ds, epochs=30, callbacks=callbacks)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    model.save(output / "vgg16_soil_moisture.keras")
    write_json(output / "metrics.json", {"test_loss": float(loss), "test_accuracy": float(accuracy), "epochs_ran": len(history.history["loss"]), "classes": classes})
    print(f"Saved model and evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
