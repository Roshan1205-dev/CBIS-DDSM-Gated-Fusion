from pathlib import Path
import os
import json
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash

ROOT = Path(os.environ.get("CBIS_ROOT", Path(__file__).resolve().parent))
FUSION_DIR = ROOT / "gated_fusion_outputs"
BASELINE_DIR = ROOT / "image_only_outputs"
MODEL_PATH = FUSION_DIR / "best_gated_fusion.keras"
PREPROCESSOR_PATH = FUSION_DIR / "metadata_preprocessor.joblib"
METRICS_PATH = FUSION_DIR / "test_metrics.json"
PREDICTIONS_PATH = FUSION_DIR / "test_predictions.csv"

IMG_SIZE = 224
CATEGORICAL_COLS = [
    "breast density", "left or right breast", "image view",
    "abnormality type", "mass shape", "mass margins",
    "calc type", "calc distribution",
]
NUMERIC_COLS = ["subtlety"]
ALL_META_COLS = CATEGORICAL_COLS + NUMERIC_COLS

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-research-dashboard-only")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

_model = None
_preprocessor = None
_startup_error = None

def load_artifacts():
    global _model, _preprocessor, _startup_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Metadata preprocessor not found: {PREPROCESSOR_PATH}")
        # The training model includes a Lambda layer; safe_mode=False is needed
        # for this locally-created, trusted checkpoint.
        _model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False, compile=False)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _startup_error = None
    except Exception as exc:
        _startup_error = f"{type(exc).__name__}: {exc}"

def read_json(path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def read_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        expected = {"patient_id", "pathology", "y_true", "probability_malignant", "y_pred"}
        if not expected.issubset(df.columns):
            return pd.DataFrame()
        df["actual"] = df["y_true"].map({0: "Benign", 1: "Malignant"})
        df["predicted"] = df["y_pred"].map({0: "Benign", 1: "Malignant"})
        df["correct"] = df["y_true"].astype(int) == df["y_pred"].astype(int)
        df["probability_pct"] = (pd.to_numeric(df["probability_malignant"], errors="coerce") * 100).round(1)
        return df
    except Exception:
        return pd.DataFrame()

def get_overview():
    metrics = read_json(METRICS_PATH)
    pred = read_predictions()
    if not pred.empty:
        y = pred["y_true"].astype(int).to_numpy()
        p = pred["y_pred"].astype(int).to_numpy()
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tp = int(((y == 1) & (p == 1)).sum())
        accuracy = float((y == p).mean())
        malignant_recall = tp / (tp + fn) if tp + fn else 0
        malignant_precision = tp / (tp + fp) if tp + fp else 0
        benign_recall = tn / (tn + fp) if tn + fp else 0
        f1 = 2 * malignant_precision * malignant_recall / (malignant_precision + malignant_recall) if malignant_precision + malignant_recall else 0
        metrics.update({
            "test_rows": len(pred),
            "test_patients": int(pred["patient_id"].nunique()),
            "accuracy_calculated": accuracy,
            "malignant_recall": malignant_recall,
            "malignant_precision": malignant_precision,
            "benign_recall": benign_recall,
            "malignant_f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "correct": int((y == p).sum()),
            "incorrect": int((y != p).sum()),
        })
    baseline = read_json(BASELINE_DIR / "test_metrics.json")
    return metrics, baseline, pred

@app.route("/")
def index():
    metrics, baseline, pred = get_overview()
    return render_template(
        "index.html",
        metrics=metrics,
        baseline=baseline,
        predictions=pred.head(704).to_dict(orient="records") if not pred.empty else [],
        prediction_count=len(pred),
        model_ready=_model is not None and _preprocessor is not None,
        startup_error=_startup_error,
        categorical_cols=CATEGORICAL_COLS,
    )

@app.route("/predict", methods=["POST"])
def predict():
    if _model is None or _preprocessor is None:
        flash("Model artifacts did not load. Check the paths and terminal error.", "error")
        return redirect(url_for("index"))
    upload = request.files.get("image")
    if not upload or not upload.filename:
        flash("Choose a JPEG or PNG crop image first.", "error")
        return redirect(url_for("index"))
    try:
        image = Image.open(upload.stream).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        image_arr = np.asarray(image, dtype=np.float32)[None, ...]  # keep 0..255 range
        row = {}
        for col in CATEGORICAL_COLS:
            row[col] = request.form.get(col, "").strip() or np.nan
        try:
            row["subtlety"] = float(request.form.get("subtlety", "").strip())
        except (TypeError, ValueError):
            row["subtlety"] = np.nan
        meta_df = pd.DataFrame([row], columns=ALL_META_COLS)
        transformed = _preprocessor.transform(meta_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)
        prob = float(np.asarray(_model.predict({"image": image_arr, "metadata": transformed}, verbose=0)).reshape(-1)[0])
        threshold = float(read_json(METRICS_PATH).get("threshold", 0.5))
        label = "Malignant" if prob >= threshold else "Benign"
        return render_template(
            "index.html",
            metrics=get_overview()[0],
            baseline=get_overview()[1],
            predictions=get_overview()[2].head(704).to_dict(orient="records") if not get_overview()[2].empty else [],
            prediction_count=len(get_overview()[2]),
            model_ready=True,
            startup_error=None,
            categorical_cols=CATEGORICAL_COLS,
            prediction_result={"label": label, "probability": prob, "threshold": threshold},
            form_values=row,
        )
    except Exception as exc:
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("index"))

@app.errorhandler(413)
def too_large(_):
    flash("Image too large. Maximum upload size is 20 MB.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Set memory growth before model load when possible.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    load_artifacts()
    app.run(host="127.0.0.1", port=5000, debug=False)
