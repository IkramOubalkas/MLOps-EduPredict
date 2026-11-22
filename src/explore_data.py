import pandas as pd

df = pd.read_csv("data/raw/students.csv", sep=";")

print("\n=== DIMENSIONS ===")
print(df.shape)

print("\n=== COLONNES ===")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n=== TYPES DE DONNEES ===")
print(df.dtypes)

print("\n=== VALEURS MANQUANTES ===")
print(df.isnull().sum().sum())

print("\n=== DOUBLONS ===")
print(df.duplicated().sum())

print("\n=== DISTRIBUTION DES CLASSES ===")
print(df["Target"].value_counts())

print("\n=== POURCENTAGES DES CLASSES ===")
print(df["Target"].value_counts(normalize=True) * 100)

print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(df.describe())
