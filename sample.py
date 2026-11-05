import pandas as pd
import json

for file in [
    "test_predictions.csv",
    "training_log.csv",
    "sample_test.csv",
]:
    try:
        df = pd.read_csv(file)
        print(f"\n--- {file} ---")
        print(df.head().to_string(index=False))
    except FileNotFoundError:
        print(f"Not found: {file}")

with open("test_metrics.json", "r") as f:
    print("\n--- test_metrics.json ---")
    print(json.dumps(json.load(f), indent=2))