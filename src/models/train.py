
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train["text"].fillna("")
y_train = train["label"]

X_test = test["text"].fillna("")
y_test = test["label"]

# Pipeline de classification
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Recherche des meilleurs paramètres
param_grid = [
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [10000, 20000],
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [10000, 20000],
        "classifier": [LinearSVC()],
        "classifier__C": [0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    }
]

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Début de l'entraînement...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nMeilleurs paramètres :")
print(grid.best_params_)

predictions = best_model.predict(X_test)

print("\nAccuracy :")
print(accuracy_score(y_test, predictions))

print("\nRapport de classification :")
print(classification_report(y_test, predictions))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, predictions))

Path("models").mkdir(exist_ok=True)

joblib.dump(best_model, "models/sentiment_pipeline.joblib")

print("\nModèle sauvegardé.")
