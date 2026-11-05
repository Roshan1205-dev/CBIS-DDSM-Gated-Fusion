from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "crop_audit_manifest.csv")

required = {"source_split", "patient_id_audit", "audit_status"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only verified, readable crops with known patient IDs.
df = df[
    (df["audit_status"] == "OK")
    & df["patient_id_audit"].notna()
].copy()

df["patient_id_audit"] = df["patient_id_audit"].astype(str).str.strip()

official_test = df[
    df["source_split"].str.endswith("_test", na=False)
].copy()

official_train = df[
    df["source_split"].str.endswith("_train", na=False)
].copy()

test_patients = set(official_test["patient_id_audit"])
overlap = test_patients & set(official_train["patient_id_audit"])

print("Verified crop rows:", len(df))
print("Official test rows:", len(official_test))
print("Overlapping patients excluded from training:", len(overlap))

# Never train on any patient represented in the official test set.
train_pool = official_train[
    ~official_train["patient_id_audit"].isin(test_patients)
].copy()

# Assign validation patients, not individual images.
patients = train_pool["patient_id_audit"].drop_duplicates().to_numpy()

if len(patients) < 5:
    raise ValueError("Too few training patients for a reliable validation split.")

train_patients, val_patients = train_test_split(
    patients,
    test_size=0.15,
    random_state=42
)

train_df = train_pool[
    train_pool["patient_id_audit"].isin(train_patients)
].copy()
val_df = train_pool[
    train_pool["patient_id_audit"].isin(val_patients)
].copy()

train_df["model_split"] = "train"
val_df["model_split"] = "validation"
official_test["model_split"] = "test"

result = pd.concat(
    [train_df, val_df, official_test],
    ignore_index=True
)

# Final leakage checks.
sets = {
    name: set(part["patient_id_audit"])
    for name, part in result.groupby("model_split")
}

assert not (sets["train"] & sets["validation"])
assert not (sets["train"] & sets["test"])
assert not (sets["validation"] & sets["test"])

out = root / "patient_safe_manifest.csv"
result.to_csv(out, index=False)

print("\nFINAL SPLIT")
for name in ["train", "validation", "test"]:
    part = result[result["model_split"] == name]
    print(
        f"{name:10s}: {len(part):4d} crop rows, "
        f"{part['patient_id_audit'].nunique():4d} patients"
    )

print("\nPatient overlap: ZERO")
print("Saved:", out)
