from pathlib import Path
import pandas as pd
from PIL import Image

root = Path("/mnt/c/Users/Admin/Music/ML")
manifest_path = root / "patient_safe_manifest.csv"
df = pd.read_csv(manifest_path)

def convert_path(value):
    if pd.isna(value) or not str(value).strip():
        return None

    s = str(value).strip().replace("\\", "/")

    # Convert absolute Windows paths into WSL paths.
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        s = f"/mnt/{drive}/" + s[2:].lstrip("/")

    p = Path(s)

    # Handle paths relative to the project root.
    if not p.is_absolute():
        p = root / p

    return str(p)

df["resolved_crop_path"] = df["resolved_crop_path"].apply(convert_path)

exists = []
readable = []

for path in df["resolved_crop_path"]:
    if not path:
        exists.append(False)
        readable.append(False)
        continue

    p = Path(path)
    exists.append(p.is_file())

    try:
        with Image.open(p) as im:
            im.verify()
        readable.append(True)
    except Exception:
        readable.append(False)

df["jpeg_exists"] = exists
df["jpeg_readable"] = readable

print("Rows:", len(df))
print("Files existing:", sum(exists))
print("Files readable:", sum(readable))
print("\nReadable by split:")
print(df.groupby("model_split")["jpeg_readable"].agg(["sum", "count"]))

bad = df[~df["jpeg_readable"]]
print("\nFirst 10 unreadable paths:")
print(bad["resolved_crop_path"].head(10).to_string(index=False))

# Preserve the original file and write a corrected manifest.
out = root / "patient_safe_manifest_fixed.csv"
df.to_csv(out, index=False)
print("\nSaved:", out)
