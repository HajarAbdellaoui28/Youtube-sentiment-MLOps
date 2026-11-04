
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("data/raw/reddit.csv")

# Adapter automatiquement les noms courants
df = df.rename(columns={
    "clean_comment": "text",
    "comment": "text",
    "category": "label",
    "sentiment": "label"
})

df = df.dropna(subset=["text", "label"])
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([-1, 0, 1])].copy()

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Distribution des sentiments
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="label",
    order=[-1, 0, 1]
)
plt.title("Distribution des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.tight_layout()
plt.savefig(output_dir / "distribution.png")
plt.show()

# 2. Longueur des commentaires
df["text_length"] = df["text"].astype(str).str.len()

plt.figure(figsize=(8, 5))
sns.histplot(df["text_length"], bins=50)
plt.title("Distribution des longueurs")
plt.xlabel("Nombre de caractères")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(output_dir / "text_length.png")
plt.show()

# 3. Statistiques descriptives
print(df["text_length"].describe())

print("\nDistribution en pourcentage :")
print(df["label"].value_counts(normalize=True) * 100)
