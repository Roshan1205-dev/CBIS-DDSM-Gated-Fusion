# CBIS-DDSM Gated Fusion

**Mammographic Lesion Classification using EfficientNetB0 and Adaptive Metadata Fusion**

A multimodal deep-learning research prototype for binary benign/malignant classification of mammographic lesion crops from the CBIS-DDSM dataset.

## Overview

This project combines mammographic image features with structured lesion metadata using a learnable gated-fusion mechanism.

The objective is to investigate whether combining visual representations and structured metadata improves classification performance compared with image-only classification.

## Model Architecture

![Gated Fusion Architecture](docs/architecture.png)

### Image Branch
- Pretrained EfficientNetB0 backbone (ImageNet)
- Input resolution: 224 × 224 × 3
- Frozen feature extractor
- Global Average Pooling
- Dense 256 + Dropout

### Metadata Branch
- 8 categorical features and lesion subtlety
- Missing-value imputation, One-Hot Encoding and scaling
- Dense 128 + Dropout
- Dense 256 feature representation

### Adaptive Gated Fusion

The two representations are combined through a learnable sigmoid gate:

\[
g=\sigma(W_g[h_i\Vert h_m]+b_g)
\]

\[
h_f=g\odot h_i+(1-g)\odot h_m
\]

The fused representation is passed through Dense 128, Dropout 0.35 and a sigmoid output for binary classification.

## Dataset

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**

The dataset contains mammographic images and associated lesion information.

- Patient-safe train, validation and test split
- ROI/cropped mammographic images
- Binary labels: Benign = 0, Malignant = 1
- Dataset is not included in this repository

## Experimental Results

Evaluation on the held-out test split:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| EfficientNetB0 (Image-only) | 0.7329 | 0.6464 |
| EfficientNetB0 + Gated Fusion | **0.8256** | **0.7280** |

The gated-fusion configuration achieved higher ROC-AUC and PR-AUC than the image-only baseline in this experiment.

These results are preliminary and require further validation, including additional fusion baselines and repeated experiments.

## Local Validation Dashboard

A Flask-based dashboard provides:

- Test metrics and confusion matrix
- Test prediction browsing and filtering
- Optional image-only baseline metrics
- Local inference using the trained gated-fusion model
- Image upload with metadata input

The uploaded image is processed in memory and is not saved by the application.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CBIS-DDSM-Gated-Fusion.git
cd CBIS-DDSM-Gated-Fusion

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\activate
```

## Model Download

Download the trained model and preprocessing artifacts from:

**[GitHub Releases](https://github.com/YOUR_USERNAME/CBIS-DDSM-Gated-Fusion/releases)**

Extract the required artifacts into `gated_fusion_outputs/`.

Required files:

- `best_gated_fusion.keras`
- `metadata_preprocessor.joblib`
- `test_metrics.json`
- `test_predictions.csv` (for the complete prediction dashboard)

The CBIS-DDSM dataset is not distributed with this repository.

## Running the Dashboard

Run from the project root:

```bash
python app.py
```

Open:

http://127.0.0.1:5000

For a separate dashboard directory, configure `CBIS_ROOT` to point to the directory containing the model artifacts.

## Technology Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- Scikit-learn
- NumPy / Pandas
- Flask
- Pillow

## Disclaimer

This project is an academic research prototype intended for experimentation and educational purposes.

It is **not a clinically validated diagnostic system** and must not be used for medical diagnosis or clinical decision-making.

## License

Specify an appropriate license before reuse or redistribution.