
from pathlib import Path
import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------
ROOT = Path("/mnt/c/Users/Admin/Music/ML")
MANIFEST = ROOT / "patient_safe_manifest_fixed.csv"
OUT = ROOT / "gated_fusion_outputs"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42

OUT.mkdir(exist_ok=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Keep GPU allocation conservative.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ---------------- LOAD DATA ----------------
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

train_df = df[df["model_split"] == "train"].copy()
val_df = df[df["model_split"] == "validation"].copy()
test_df = df[df["model_split"] == "test"].copy()

assert len(train_df) and len(val_df) and len(test_df)
assert set(train_df.patient_id_audit).isdisjoint(val_df.patient_id_audit)
assert set(train_df.patient_id_audit).isdisjoint(test_df.patient_id_audit)
assert set(val_df.patient_id_audit).isdisjoint(test_df.patient_id_audit)

print("\nRows:", len(train_df), len(val_df), len(test_df))
print("Training label counts:")
print(train_df["label"].value_counts().sort_index())

# ---------------- METADATA ----------------
# Do not use patient IDs, image paths, assessment, or pathology as inputs.
categorical_cols = [
    "breast density",
    "left or right breast",
    "image view",
    "abnormality type",
    "mass shape",
    "mass margins",
    "calc type",
    "calc distribution",
]
numeric_cols = ["subtlety"]

for c in categorical_cols:
    if c not in df.columns:
        df[c] = np.nan

for c in numeric_cols:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Fit all metadata preprocessing on TRAIN only.
metadata_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ],
    remainder="drop",
)

X_train_meta = metadata_preprocessor.fit_transform(train_df)
X_val_meta = metadata_preprocessor.transform(val_df)
X_test_meta = metadata_preprocessor.transform(test_df)

# TensorFlow model input is dense float32.
from scipy import sparse

def to_dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)

X_train_meta = to_dense_float32(X_train_meta)
X_val_meta = to_dense_float32(X_val_meta)
X_test_meta = to_dense_float32(X_test_meta)
joblib.dump(metadata_preprocessor, OUT / "metadata_preprocessor.joblib")
print("Metadata dimensions:", X_train_meta.shape[1])

# ---------------- IMAGE PIPELINE ----------------
def decode_image(path, label, metadata):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)  # EfficientNet input range: 0..255
    return {"image": image, "metadata": metadata}, tf.cast(label, tf.float32)

def make_dataset(frame, metadata, training=False):
    paths = frame["resolved_crop_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, metadata))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, X_train_meta, training=True)
val_ds = make_dataset(val_df, X_val_meta)
test_ds = make_dataset(test_df, X_test_meta)

# ---------------- GATED FUSION MODEL ----------------
image_input = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3), name="image"
)
metadata_input = tf.keras.Input(
    shape=(X_train_meta.shape[1],), name="metadata"
)

# Frozen image backbone for the first experiment.
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
)
backbone.trainable = False

image_features = backbone(image_input, training=False)
image_features = tf.keras.layers.GlobalAveragePooling2D()(image_features)
image_features = tf.keras.layers.Dense(
    256, activation="relu", name="image_projection"
)(image_features)
image_features = tf.keras.layers.Dropout(0.25)(image_features)

metadata_features = tf.keras.layers.Dense(
    128, activation="relu"
)(metadata_input)
metadata_features = tf.keras.layers.Dropout(0.20)(metadata_features)
metadata_features = tf.keras.layers.Dense(
    256, activation="relu", name="metadata_projection"
)(metadata_features)

# Element-wise learned gate.
combined = tf.keras.layers.Concatenate()(
    [image_features, metadata_features]
)
gate = tf.keras.layers.Dense(
    256, activation="sigmoid", name="fusion_gate"
)(combined)

image_weighted = tf.keras.layers.Multiply()(
    [gate, image_features]
)
metadata_weighted = tf.keras.layers.Lambda(
    lambda x: (1.0 - x[0]) * x[1],
    name="metadata_gate_complement",
)([gate, metadata_features])

fused = tf.keras.layers.Add()(
    [image_weighted, metadata_weighted]
)
fused = tf.keras.layers.Dense(128, activation="relu")(fused)
fused = tf.keras.layers.Dropout(0.35)(fused)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy")(fused)

model = tf.keras.Model(
    inputs={"image": image_input, "metadata": metadata_input},
    outputs=output,
    name="CBIS_DDSM_Gated_Fusion",
)

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

# Weights based on training labels only.
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
        str(OUT / "best_gated_fusion.keras"),
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

model.save(OUT / "final_gated_fusion.keras")

# ---------------- HELD-OUT TEST ----------------
# Use the best validation-selected checkpoint for final evaluation.
best_model = tf.keras.models.load_model(
    OUT / "best_gated_fusion.keras",
    safe_mode=False,
)

probabilities = best_model.predict(test_ds, verbose=1).ravel()
y_true = test_df["label"].to_numpy(dtype=int)
y_pred = (probabilities >= 0.5).astype(int)

print("\nTEST CLASSIFICATION REPORT")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    digits=4,
    zero_division=0,
))

print("Confusion matrix [benign, malignant]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("Test ROC-AUC:", round(roc_auc_score(y_true, probabilities), 4))
print("Test PR-AUC:", round(average_precision_score(y_true, probabilities), 4))

with open(OUT / "test_metrics.json", "w") as f:
    json.dump({
        "test_roc_auc": float(roc_auc_score(y_true, probabilities)),
        "test_pr_auc": float(average_precision_score(y_true, probabilities)),
        "test_rows": int(len(test_df)),
        "test_patients": int(test_df["patient_id_audit"].nunique()),
        "threshold": 0.5,
    }, f, indent=2)

pd.DataFrame({
    "patient_id": test_df["patient_id_audit"].to_numpy(),
    "pathology": test_df["pathology"].to_numpy(),
    "y_true": y_true,
    "probability_malignant": probabilities,
    "y_pred": y_pred,
}).to_csv(OUT / "test_predictions.csv", index=False)

print("\nSaved outputs to:", OUT)
