from pathlib import Path
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "patient_safe_manifest.csv")

print("Columns:", df.columns.tolist())

for col in ["pathology", "resolved_crop_path", "jpeg_exists", "model_split"]:
    if col not in df.columns:
        print(f"\nMISSING REQUIRED COLUMN: {col}")

print("\nPathology distribution by split:")
print(pd.crosstab(df["model_split"], df["pathology"], margins=True))

print("\nLabel values:")
print(df["pathology"].value_counts(dropna=False).to_string())

# Verify a sample of paths before attempting full image loading.
path_col = "resolved_crop_path"
sample = df[df[path_col].notna()].sample(
    min(20, df[path_col].notna().sum()), random_state=42
)

ok = 0
for _, row in sample.iterrows():
    p = Path(str(row[path_col]))
    if not p.is_absolute():
        p = root / p
    try:
        with Image.open(p) as im:
            im.verify()
        ok += 1
    except Exception as e:
        print("BAD IMAGE:", p, "|", e)

print(f"\nReadable image sample: {ok}/{len(sample)}")
