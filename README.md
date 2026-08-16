# Soil Moisture Edge

Reconstructed, open-source implementation for **"Real-Time Non-Contact Soil Moisture Estimation Using Machine Learning at the Edge."** It estimates three image-based moisture classes under controlled lighting:

| Class | Volumetric water content |
| --- | --- |
| `dry` | < 35% |
| `optimum` | 35% to 50%, inclusive |
| `saturated` | > 50% |

The original research code was unavailable. This repository was rebuilt from the paper, so it is a faithful methodological implementation - not an assertion of bit-for-bit reproduction of the reported results. In particular, exact sample splits, image filenames, random seeds, and preprocessing details not reported in the paper cannot be recovered.

## What is included

- **VGG16** transfer-learning baseline: frozen ImageNet convolutional base, 128 x 128 input, 256-unit L2-regularized head, dropout 0.5, Adam, early stopping, and learning-rate reduction.
- **Random Forest regressor**: 100 trees trained on eight color/gray-histogram features and continuous moisture percentage.
- **RBF SVM**: standardized, normalized RGB-histogram features (`C=1`, `gamma='scale'`), with optional PCA for cross-validation.
- Raspberry Pi webcam inference for a saved SVM pipeline.

## Setup

```bash
git clone https://github.com/<your-org>/soil-moisture-edge.git
cd soil-moisture-edge
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For the VGG16 baseline, also install a TensorFlow build appropriate for your machine. The base requirements intentionally keep edge/classical usage lightweight.

## Dataset layout

Download the INSA soil image dataset cited in the paper from [Zenodo](https://zenodo.org/records/13322242), then arrange or symlink its RGB images below. `src.preprocessing.labels` derives the soil type and moisture percentage from folder or filename tokens such as `Kaolin/45/` or `Silt_45`.

```text
data/raw/
  Kaolin/
    30/
      image_01.jpg
  Silt/
    45/
      image_01.jpg
```

If your archive uses a different convention, create `data/labels.csv` with `path,soil_type,moisture_pct`; paths are relative to `data/raw`. The CSV takes precedence.

## Train and evaluate

```bash
# Train a regressor for continuous water-content prediction.
python -m src.training.train_random_forest --data-dir data/raw --output-dir artifacts/random_forest

# Train the moisture-class SVM and save a Raspberry Pi-ready pipeline.
python -m src.training.train_svm --data-dir data/raw --output-dir artifacts/svm --cross-validate --pca-components 100

# Fine-tune the VGG16 baseline (requires TensorFlow).
python -m src.training.train_vgg16 --data-dir data/raw --output-dir artifacts/vgg16
```

Each command writes its fitted model, metrics JSON, and a labeled confusion-matrix PNG. The SVM uses a stratified 80/20 split and evaluates optional 5-fold macro-F1 cross-validation separately. With only 150 controlled-lab images, reported scores should be treated as laboratory estimates, not field performance.

## Raspberry Pi inference

```bash
python deployment/raspberry_pi.py \
  --model artifacts/svm/svm_pipeline.joblib \
  --camera-index 0
```

Press `q` to exit. For consistent results, use a fixed camera position and stable overhead illumination, as in the study. The script uses OpenCV's standard VideoCapture interface; it works with a USB webcam and does not require cloud connectivity.

## Reproducibility notes

- Default seed: `42`; adjust settings in `configs/default.yaml`.
- The paper contains an apparent label-description inconsistency: it mentions composite soil-and-moisture folders but specifies a three-unit output and reports three moisture classes. This implementation trains the classifiers on the three moisture bands; soil type is retained as metadata.
- No dataset images, trained weights, or claimed paper-result artifacts are redistributed here.

## License

Code is released under the MIT License. Dataset licensing remains with its original authors and host.
