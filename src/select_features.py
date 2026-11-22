import pandas as pd
import yaml

with open("params.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["raw_path"], sep=";")
df.columns = df.columns.str.strip()

numerical = config["features"]["numerical"]
categorical = config["features"]["categorical"]

features = numerical + categorical
target = config["data"]["target"]

missing = [col for col in features + [target]
           if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

assert not any("2nd sem" in col for col in features)

X = df[features]
y = df[target]

print("Original dataset:", df.shape)
print("Selected features:", len(features))
print("Numerical features:", len(numerical))
print("Categorical features:", len(categorical))
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\n=== SELECTED FEATURES ===")
for col in features:
    print("-", col)

print("\nSUCCESS: Feature selection completed.")
