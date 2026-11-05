
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

model = joblib.load(MODEL_PATH)

# Cas limites
comments = [
    "Good",
    "Bad",
    "a",
    "This is amazing! 😍😍😍",
    "I hate this video 😡",
    "C'est magnifique, I love it!",
    "Very interesting explanation. " * 100,
    "123456789",
]

print("\n=== TESTS DES CAS LIMITES ===")

start = time.perf_counter()
predictions = model.predict(comments)
elapsed = time.perf_counter() - start

for text, prediction in zip(comments, predictions):
    print(f"{text[:65]} -> {prediction}")

print(f"\nTemps total : {elapsed:.4f} s")
print(f"Temps moyen : {elapsed / len(comments) * 1000:.2f} ms/commentaire")

# Evaluation sur le jeu de test
if TEST_PATH.exists():
    df = pd.read_csv(TEST_PATH)

    print("\nColonnes disponibles :", df.columns.tolist())
    print("Adapte TEXT_COL et LABEL_COL aux noms affichés.")

    TEXT_COL = "clean_text"
    LABEL_COL = "label"

    if TEXT_COL in df.columns and LABEL_COL in df.columns:
        df = df.dropna(subset=[TEXT_COL, LABEL_COL])
        y_pred = model.predict(df[TEXT_COL].astype(str))

        print("\nAccuracy :", accuracy_score(df[LABEL_COL], y_pred))
        print("\nClassification report :")
        print(classification_report(df[LABEL_COL], y_pred, zero_division=0))
        print("\nMatrice de confusion :")
        print(confusion_matrix(df[LABEL_COL], y_pred))
    else:
        print("Modifie les noms de colonnes dans le script.")
else:
    print("Fichier test.csv introuvable.")
