
import pandas as pd

df = pd.read_csv("data/raw/reddit.csv")

print("Dimensions :", df.shape)

print("\nColonnes :")
print(df.columns.tolist())

print("\nPremières lignes :")
print(df.head())

print("\nTypes des colonnes :")
print(df.dtypes)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nDistribution des labels :")
if "label" in df.columns:
    print(df["label"].value_counts())
else:
    print("Identifier d'abord la colonne des sentiments.")
