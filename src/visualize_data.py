from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/students.csv", sep=";")
df.columns = df.columns.str.strip()

output = Path("reports/eda")
output.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

colors = {
    "Graduate": "#2E8B57",
    "Dropout": "#DC3545",
    "Enrolled": "#E5A823"
}

# 1. Target distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df, x="Target", hue="Target",
    palette=colors, legend=False
)
plt.title("Academic Status Distribution")
plt.xlabel("Academic Status")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(output / "01_target_distribution.png", dpi=150)
plt.close()

# 2. First semester grades
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df, x="Target",
    y="Curricular units 1st sem (grade)",
    hue="Target", palette=colors, legend=False
)
plt.title("First Semester Grades by Academic Status")
plt.tight_layout()
plt.savefig(output / "02_semester_grades.png", dpi=150)
plt.close()

# 3. Approved curricular units
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df, x="Target",
    y="Curricular units 1st sem (approved)",
    hue="Target", palette=colors, legend=False
)
plt.title("Approved Units by Academic Status")
plt.tight_layout()
plt.savefig(output / "03_approved_units.png", dpi=150)
plt.close()

# 4. Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, x="Age at enrollment",
    hue="Target", palette=colors,
    bins=30, multiple="stack"
)
plt.title("Age Distribution at Enrollment")
plt.tight_layout()
plt.savefig(output / "04_age_distribution.png", dpi=150)
plt.close()

# 5. Academic correlations
columns = [
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (evaluations)"
]

plt.figure(figsize=(12, 9))
sns.heatmap(
    df[columns].corr(),
    annot=True, cmap="coolwarm",
    fmt=".2f", vmin=-1, vmax=1
)
plt.title("Academic Features Correlation Matrix")
plt.tight_layout()
plt.savefig(output / "05_correlation_matrix.png", dpi=150)
plt.close()

print("SUCCESS: 5 EDA charts generated.")
print(f"Output directory: {output}")
