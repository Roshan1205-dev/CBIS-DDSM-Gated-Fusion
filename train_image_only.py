#!/usr/bin/env python3
"""Image-only CBIS-DDSM baseline for comparison with the gated-fusion model."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

# ---------------- CONFIG ----------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8

ROOT = Path("/mnt/c/Users/Admin/Music/ML")
MANIFEST = ROOT / "patient_safe_manifest_fixed.csv"
OUT = ROOT / "image_only_outputs"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Conservative GPU memory allocation.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Output directory:", OUT)

# ---------------- LOAD MANIFEST / LABELS ----------------
df = pd.read_csv(MANIFEST)
df = df[df["jpeg_readable"].astype(str).str.lower().isin(["true", "1"])].copy()

df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
df["label"] = df["pathology"].map({
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
})
df = df.dropna(subset=["label", "resolved_crop_path", "patient_id_audit"])
df["label"] = df["label"].astype("int32")

train_df = df[df["model_split"] == "train"].copy().reset_index(drop=True)
val_df = df[df["model_split"] == "validation"].copy().reset_index(drop=True)
test_df = df[df["model_split"] == "test"].copy().reset_index(drop=True)

assert len(train_df) and len(val_df) and len(test_df), "One or more splits are empty."
assert set(train_df.patient_id_audit).isdisjoint(val_df.patient_id_audit)
assert set(train_df.patient_id_audit).isdisjoint(test_df.patient_id_audit)
assert set(val_df.patient_id_audit).isdisjoint(test_df.patient_id_audit)

print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
print("Training label counts:")
print(train_df["label"].value_counts().sort_index())

# ---------------- IMAGE PIPELINE ----------------
# Keep pixel values in the 0..255 range, matching the fusion experiment's
# EfficientNetB0 input pipeline.
def decode_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)

def make_dataset(frame, training=False):
    paths = frame["resolved_crop_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ---------------- IMAGE-ONLY MODEL ----------------
image_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
)
backbone.trainable = False

x = backbone(image_input, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x = tf.keras.layers.Dense(256, activation="relu", name="image_features")(x)
x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy_probability")(x)

model = tf.keras.Model(inputs=image_input, outputs=output, name="cbis_ddsm_image_only")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
    ],
)
model.summary()

classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].to_numpy(),
)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weights)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(OUT / "best_image_only.keras"),
        monitor="val_roc_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_roc_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(OUT / "training_log.csv")),
    tf.keras.callbacks.TerminateOnNaN(),
]

# ---------------- TRAIN ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)
model.save(OUT / "final_image_only.keras")

# ---------------- HELD-OUT TEST ----------------
# Reload the best checkpoint selected using validation ROC-AUC.
best_model = tf.keras.models.load_model(OUT / "best_image_only.keras")
probabilities = best_model.predict(test_ds, verbose=1).ravel()
y_true = test_df["label"].to_numpy(dtype=int)
y_pred = (probabilities >= 0.5).astype(int)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["benign", "malignant"],
    output_dict=True,
    zero_division=0,
)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics = {
    "model": "image_only_EfficientNetB0_frozen",
    "seed": SEED,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "accuracy_at_threshold_0_5": float(accuracy_score(y_true, y_pred)),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
    "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
    "confusion_matrix_labels_benign_malignant": cm.tolist(),
    "classification_report": report,
}

print("\nTEST CLASSIFICATION REPORT")
print(classification_report(
    y_true, y_pred, labels=[0, 1],
    target_names=["benign", "malignant"], zero_division=0
))
print("Confusion matrix (rows=true, columns=predicted; [benign, malignant]):")
print(cm)
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Test average precision (PR-AUC summary): {metrics['pr_auc_average_precision']:.4f}")

with open(OUT / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

predictions = test_df[[
    "patient id", "patient_id_audit", "resolved_crop_path", "pathology", "label"
]].copy()
predictions["malignant_probability"] = probabilities
predictions["predicted_label_threshold_0_5"] = y_pred
predictions.to_csv(OUT / "test_predictions.csv", index=False)

print("\nSaved outputs to:", OUT)
print(" - best_image_only.keras")
print(" - final_image_only.keras")
print(" - training_log.csv")
print(" - test_metrics.json")
print(" - test_predictions.csv")
