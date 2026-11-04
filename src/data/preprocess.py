
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/reddit.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Adapter les colonnes au format demandé
text_candidates = ["text", "comment", "clean_comment"]
label_candidates = ["label", "category", "sentiment"]

text_col = next(
    (c for c in text_candidates if c in df.columns),
    None
)
label_col = next(
    (c for c in label_candidates if c in df.columns),
    None
)

if text_col is None or label_col is None:
    raise ValueError(
        f"Colonnes non reconnues : {df.columns.tolist()}"
    )

df = df.rename(columns={
    text_col: "text",
    label_col: "label"
})

df = df[["text", "label"]].copy()

# Supprimer les valeurs manquantes
df = df.dropna(subset=["text", "label"])

# Convertir les labels en nombres
df["label"] = pd.to_numeric(
    df["label"], errors="coerce"
)
df = df.dropna(subset=["label"])
df = df[df["label"].isin([-1, 0, 1])].copy()
df["label"] = df["label"].astype(int)

def clean_text(text):
    text = str(text).lower()

    # Supprimer les URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Supprimer les mentions
    text = re.sub(r"@\w+", " ", text)

    # Supprimer les caractères spéciaux
    # Conserver les emojis pour une étude ultérieure
    text = re.sub(r"[^\w\s!?']", " ", text)

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["text"] = df["text"].apply(clean_text)

# Supprimer les textes vides
df = df[df["text"].str.len() > 0]

# Supprimer les doublons de texte
df = df.drop_duplicates(subset=["text"])

print("Distribution après nettoyage :")
print(df["label"].value_counts().sort_index())

# Vérifier le minimum demandé
counts = df["label"].value_counts()
if not all(counts.get(c, 0) >= 300 for c in [-1, 0, 1]):
    raise ValueError(
        "Dataset insuffisant : au moins 300 textes par classe."
    )

# Séparation reproductible
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nEnsemble train :", train_df.shape)
print("Ensemble test :", test_df.shape)
print("Préparation terminée.")
