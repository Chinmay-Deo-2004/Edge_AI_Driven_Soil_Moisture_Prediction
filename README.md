<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/51567fbb-e652-4024-9b97-33e7b5aabb57" />

<table> 

<tr>
  <td><b>Technologies</b></td>
  <td>
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit--learn&logoColor=white">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white">
  </td>
</tr>

<tr>
  <td><b>Hardware</b></td>
  <td>
    <img src="https://img.shields.io/badge/Raspberry%20Pi%205-A22846?logo=raspberrypi&logoColor=white">
  </td>
</tr>

<tr> <td><b>Dataset</b></td> <td> <a href="https://zenodo.org/records/13322242"> <img src="https://img.shields.io/badge/INSA%20Soil%20Image%20Dataset-Zenodo-2E86C1"> </a> </td> </tr>

<tr> <td><b>Publication</b></td> <td> <a href="https://doi.org/10.1109/ICEdge67252.2025.11412732"> <img src="https://img.shields.io/badge/DOI-10.1109%2FICEdge67252.2025.11412732-007EC6"> </a> </td> </tr>

<tr> <td><b>License</b></td> <td> <img src="https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square"> </td> </tr> </table>

## Project Structure

```text
.
│
├── src/
│   ├── preprocessing/   Image preprocessing and feature extraction
│   ├── models/          VGG16, Random Forest, and SVM implementations
│   ├── training/        Model training pipelines
│   ├── evaluation/      Metrics, confusion matrices, and cross-validation
│   └── utils/            Shared utilities
│
├── deployment/
│   └── raspberry_pi.py  Raspberry Pi webcam inference
│
├── configs/
│   └── default.yaml     Configuration and reproducibility settings
│
├── README.md            Project documentation
├── requirements.txt     Python dependencies
├── LICENSE              MIT License
└── CITATION.cff         Citation metadata

```
## Setup

Clone the repository:

```bash
git clone https://github.com/<your-org>/soil-moisture-edge.git
cd soil-moisture-edge
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For the VGG16 baseline, also install a TensorFlow build appropriate for your machine. The base requirements intentionally keep classical-ML and edge-deployment usage lightweight.

## Dataset

Download the **INSA Soil Image Dataset** cited in the paper from [Zenodo](https://zenodo.org/records/13322242).

The dataset is **not included or redistributed in this repository**.

The paper describes 150 images in total, consisting of 70 silt images and 80 kaolin images. The images were captured under controlled laboratory conditions using a Luxonis OAK-D Pro RGB camera.

Provide the downloaded dataset path to the training commands below.

## Train and Evaluate

### Random Forest

Train the Random Forest regressor for continuous water-content prediction:

```bash
python -m src.training.train_random_forest \
    --data-dir /path/to/insa-dataset \
    --output-dir artifacts/random_forest
```

The Random Forest pipeline extracts eight handcrafted features from each image:

- Mean R, G, and B intensities
- Mean HSV value
- Mean HSL lightness
- Mean CIEXYZ Z-channel intensity
- Mean and standard deviation of a Gaussian fit to the grayscale histogram

The resulting feature vector is used to predict continuous volumetric water content. The Random Forest consists of **100 decision trees**.

The continuous predictions can subsequently be mapped to the three moisture categories using the following thresholds:

| Class | Volumetric water content |
| --- | --- |
| `dry` | < 35% |
| `optimum` | 35–50% |
| `saturated` | > 50% |

### SVM

Train the RBF-kernel SVM moisture classifier:

```bash
python -m src.training.train_svm \
    --data-dir /path/to/insa-dataset \
    --output-dir artifacts/svm
```

The SVM pipeline converts each image into a three-dimensional RGB color histogram, normalizes and flattens the histogram, and standardizes the resulting feature vector before classification.

The classifier uses:

```text
Kernel: RBF
C: 1
gamma: scale
```

### SVM + PCA

The paper also evaluates five-fold cross-validation with and without PCA:

```bash
python -m src.training.train_svm \
    --data-dir /path/to/insa-dataset \
    --output-dir artifacts/svm \
    --cross-validate \
    --pca-components 100
```

The PCA configuration uses **100 components**.

### VGG16

Fine-tune the VGG16 classification baseline:

```bash
python -m src.training.train_vgg16 \
    --data-dir /path/to/insa-dataset \
    --output-dir artifacts/vgg16
```

Images are resized to **128 × 128 pixels** and normalized to `[0, 1]`.

Training augmentation includes:

- Random rotations up to ±20°
- Horizontal flipping
- Random zooming up to ±20%
- Width and height shifts up to 20%

The architecture uses a frozen ImageNet-pretrained VGG16 convolutional base followed by:

```text
Input: 128 × 128 × 3
        ↓
Frozen VGG16 convolutional base
        ↓
Flatten
        ↓
Dense(256, ReLU, L2 regularization)
        ↓
Dropout(0.5)
        ↓
Dense(3, Softmax)
```

The paper specifies Adam optimization, categorical cross-entropy, a batch size of 16, training for up to 30 epochs, early stopping, and `ReduceLROnPlateau`.

## Raspberry Pi Inference

The best-performing SVM model can be deployed for offline webcam-based inference:

```bash
python deployment/raspberry_pi.py \
    --model artifacts/svm/svm_pipeline.joblib \
    --camera-index 0
```

Press `q` to exit.

The experimental deployment used a **Raspberry Pi 5 with 4 GB RAM**, Raspberry Pi AI Kit, and a USB webcam mounted under controlled overhead illumination.

For conditions comparable to the reported experiment, maintain a fixed camera position and stable overhead illumination.

The system performs inference locally and does not require cloud connectivity or external sensing hardware.

## Results

The paper reports the following performance on the controlled laboratory dataset:

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| SVM | 100%* | 1.00* | 1.00* |
| VGG16 | 47% | 0.32 | 0.41 |
| Random Forest | 97% | 0.92 | 0.97 |

\* The reported SVM result was obtained on the limited laboratory dataset and may not generalize to diverse field conditions.

The Random Forest regression experiment additionally reported:

| Metric | Value |
| --- | ---: |
| MSE | 52.69 |
| R² | 0.917 |

After binning the continuous predictions into the three moisture classes, the reported classification accuracy was **97%**.

### SVM Five-Fold Cross-Validation

| Configuration | Fold F1-scores | Mean F1 |
| --- | --- | ---: |
| Without PCA | 1.00, 0.69, 0.44, 0.61, 0.21 | 0.59 |
| With PCA | 1.00, 1.00, 0.66, 1.00, 0.54 | 0.84 |

The paper reports that PCA improved the mean F1 score and reduced fold-to-fold variance.

## Citation

If you use this implementation or build upon this work, please cite:

```bibtex
@INPROCEEDINGS{11412732,
  author={Singh, Chhavi and Deo, Chinmay and Pahuja, Roop},
  booktitle={2025 First International Conference on Intelligent Computing and Systems at the Edge (ICEdge)},
  title={Real-Time Non-Contact Soil Moisture Estimation Using Machine Learning at the Edge},
  year={2025},
  doi={10.1109/ICEdge67252.2025.11412732}
}
```
