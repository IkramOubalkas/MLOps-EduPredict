from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["raw_path"], sep=";")
df.columns = df.columns.str.strip()

target = config["data"]["target"]

numerical = config["features"]["numerical"]
categorical = config["features"]["categorical"]
features = numerical + categorical

required = features + [target]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if any("2nd sem" in col for col in features):
    raise ValueError("Data leakage: second semester features found.")

dataset = df[required].copy()

seed = config["data"]["random_state"]
test_size = config["data"]["test_size"]
val_size = config["data"]["validation_size"]

# Hold out the final test set.
train_val, test = train_test_split(
    dataset,
    test_size=test_size,
    random_state=seed,
    stratify=dataset[target]
)

# Validation size relative to the remaining data.
relative_val_size = val_size / (1 - test_size)

train, val = train_test_split(
    train_val,
    test_size=relative_val_size,
    random_state=seed,
    stratify=train_val[target]
)

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

train.to_csv(output / "train.csv", index=False)
val.to_csv(output / "val.csv", index=False)
test.to_csv(output / "test.csv", index=False)

print("=== DATA SPLIT ===")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

print("\n=== CLASS DISTRIBUTION (%) ===")
for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    print(f"\n{name}:")
    print(
        (part[target].value_counts(normalize=True) * 100)
        .round(2)
    )

print("\nSUCCESS: Data preprocessing completed.")
