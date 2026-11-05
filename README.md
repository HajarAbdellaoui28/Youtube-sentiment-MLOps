# Analyse de sentiment des commentaires YouTube — Projet MLOps

## 1. Présentation du projet

L'objectif est de développer une solution permettant d'analyser automatiquement les sentiments exprimés dans les commentaires YouTube grâce à un modèle de Machine Learning, une API REST FastAPI, une extension Chrome et un déploiement Docker sur GitHub Codespaces.

### Problématique

L'analyse manuelle de nombreux commentaires YouTube est chronophage, subjective et difficile à quantifier. La solution proposée classe les commentaires en trois catégories : **positif**, **neutre** et **négatif**, puis présente des statistiques synthétiques dans une extension Chrome.

## 2. Objectifs

1. Préparer et explorer un jeu de données textuelles.
2. Entraîner et optimiser des modèles de classification de sentiment.
3. Évaluer le modèle sur un jeu de test indépendant.
4. Exposer le modèle avec une API FastAPI prenant en charge les lots de commentaires.
5. Développer une extension Chrome pour analyser les commentaires YouTube.
6. Conteneuriser l'API avec Docker et la déployer sur GitHub Codespaces.
7. Effectuer des tests du modèle, de l'API, de l'extension et du flux complet.

## 3. Architecture technique

```text
Utilisateur → Vidéo YouTube → Extension Chrome
                                  |
                                  v
                           API REST FastAPI
                         POST /predict_batch
                                  |
                                  v
                     Pipeline TF-IDF + classificateur
                                  |
                                  v
                     Prédictions et statistiques
                                  |
                                  v
                         Extension Chrome
```

- **Frontend :** extension Chrome (HTML, CSS, JavaScript, Manifest V3).
- **Backend :** API REST FastAPI avec validation Pydantic.
- **Machine Learning :** pipeline scikit-learn (TF-IDF et classificateur supervisé).
- **Infrastructure :** conteneur Docker exécuté localement ou dans GitHub Codespaces.

## 4. Technologies

| Domaine | Outils |
|---|---|
| Langage principal | Python 3.12 |
| Données | Pandas, NumPy |
| Machine Learning | scikit-learn, TF-IDF, Logistic Regression, Linear SVM |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Extension Chrome | JavaScript, HTML, CSS, Manifest V3 |
| Conteneurisation | Docker |
| Cloud | GitHub Codespaces |
| Tests | Pytest, HTTPX |
| Versioning | Git, GitHub |
| Sérialisation | Joblib |

## 5. Structure du projet

```text
Youtube-sentiment-MLOps/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   └── api/
│       └── main.py
├── models/
│   └── sentiment_pipeline.joblib
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── benchmark_api.py
├── chrome-extension/
│   ├── manifest.json
│   ├── content.js
│   ├── popup.html
│   ├── popup.css
│   └── popup.js
├── app_api.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-prod.txt
└── README.md
```

## 6. Données et prétraitement

Le projet exploite un jeu de données **Reddit** contenant des textes annotés pour l'analyse de sentiment.

| Étiquette | Classe |
|---|---|
| `-1` | Négatif |
| `0` | Neutre |
| `1` | Positif |

Les étapes comprennent l'inspection des données, la gestion des valeurs manquantes, le nettoyage des textes, l'analyse exploratoire, puis la séparation en ensembles d'entraînement et de test.

## 7. Entraînement et optimisation

La représentation des textes est assurée par **TF-IDF (Term Frequency–Inverse Document Frequency)**. Deux modèles linéaires sont étudiés : la **régression logistique** et le **SVM linéaire**. Une recherche d'hyperparamètres avec **GridSearchCV** permet de comparer les configurations.

Le pipeline final est enregistré dans :

```text
models/sentiment_pipeline.joblib
```

Pour relancer l'entraînement, si le script correspond à cette organisation :

```bash
python -m src.models.train
```

## 8. Évaluation du modèle

Le modèle est évalué sur un jeu de test indépendant à l'aide des métriques suivantes :

- **Accuracy :** proportion totale de prédictions correctes.
- **Precision :** pertinence des prédictions pour chaque classe.
- **Recall :** proportion des observations d'une classe correctement retrouvées.
- **F1-score :** moyenne harmonique de la précision et du rappel.
- **Matrice de confusion :** répartition des prédictions correctes et des erreurs.

| Métrique | Résultat |
|---|---|
| Accuracy | 89,94 % |
| Precision macro | 0,89 |
| Recall macro | 0,89 |
| F1-score macro | 0,89 |
| Nombre d'observations de test | 7 313 |

Des cas limites sont également testés : textes courts ou longs, emojis, chiffres et langues mélangées.

## 9. API REST FastAPI

### `GET /health`

Vérifie que l'API est disponible et que le modèle est chargé.

**Exemple de réponse :**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict_batch`

Reçoit un lot de commentaires et retourne les prédictions ainsi que leur répartition par sentiment.

**Exemple de requête :**

```json
{
  "comments": [
    "This video is amazing!",
    "I hate this content.",
    "This is an ordinary video."
  ]
}
```

**Exemple illustratif de structure de réponse :**

```json
{
  "total": 3,
  "results": [
    {"text": "This video is amazing!", "label": 1, "sentiment": "positive"},
    {"text": "I hate this content.", "label": -1, "sentiment": "negative"},
    {"text": "This is an ordinary video.", "label": 0, "sentiment": "neutral"}
  ],
  "statistics": {
    "negative": 1,
    "neutral": 1,
    "positive": 1
  }
}
```

Les prédictions réelles dépendent du modèle. Les requêtes invalides, notamment les listes vides, peuvent produire une réponse HTTP `422` grâce à la validation Pydantic.

Documentation interactive locale : `http://127.0.0.1:7860/docs` (Docker) ou `http://127.0.0.1:8000/docs` (Uvicorn par défaut).

## 10. Installation et exécution locale

### Prérequis

- Python 3.12
- Git
- Docker Desktop (pour l'exécution conteneurisée)
- Google Chrome

### Cloner le projet

```bash
git clone https://github.com/HajarAbdellaoui28/Youtube-sentiment-MLOps.git
cd Youtube-sentiment-MLOps
```

### Préparer l'environnement Python

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Lancer l'API sans Docker

```bash
python -m uvicorn src.api.main:app --reload
```

Documentation : `http://127.0.0.1:8000/docs`.

## 11. Conteneurisation Docker

Construire l'image :

```bash
docker build -t youtube-sentiment-api .
```

Démarrer l'API :

```bash
docker run --rm -p 7860:7860 youtube-sentiment-api
```

Tester :

- `http://127.0.0.1:7860/health`
- `http://127.0.0.1:7860/docs`

Le `Dockerfile` utilise une image Python légère, un fichier de dépendances de production et un `.dockerignore` pour limiter les fichiers copiés.


## 12. Déploiement sur GitHub Codespaces

1. Pousser le code, le Dockerfile et le modèle `.joblib` sur GitHub.
2. Ouvrir le dépôt dans **Code → Codespaces**.
3. Créer ou démarrer un Codespace.
4. Construire l'image et lancer le conteneur :

```bash
docker build -t youtube-sentiment-api .
docker run -d --name sentiment-api -p 7860:7860 youtube-sentiment-api
```

5. Dans l'onglet **Ports**, définir la visibilité du port **7860** sur **Public**.
6. Copier l'URL HTTPS fournie par GitHub.
7. Vérifier `/health`, `/docs` et `/predict_batch`.

**URL de l'API Cloud :** `https://orange-orbit-6945v76wjrr934wpg-7860.app.github.dev`.


## 13. Extension Chrome

L'extension récupère les commentaires disponibles sur une vidéo YouTube, les envoie à l'API par lots et présente les résultats.

### Installation

1. Ouvrir `chrome://extensions`.
2. Activer le **Mode développeur**.
3. Cliquer sur **Charger l'extension non empaquetée**.
4. Sélectionner le dossier `chrome-extension`.
5. Actualiser la page YouTube après le chargement ou une mise à jour de l'extension.

### Configuration de l'API

Dans `popup.js`, adapter l'URL utilisée par l'application :

```javascript
// API locale
const API_URL = "http://127.0.0.1:7860";
```

ou :

```javascript
// API Cloud : remplacer par l'URL réelle
const API_URL = "https://NOM-DU-CODESPACE-7860.app.github.dev";
```

### Utilisation

1. Ouvrir une vidéo YouTube comportant des commentaires.
2. Faire défiler la page pour charger les commentaires.
3. Ouvrir l'extension et lancer l'analyse.
4. Consulter la répartition des sentiments et les prédictions.
5. Utiliser les fonctionnalités effectivement implémentées (filtres, thème sombre, copie/export, etc.).

![alt text](image-1.png)


## Conclusion

Ce projet illustre une chaîne MLOps allant de la préparation des données et de l'entraînement d'un modèle de classification à son exposition par une API REST, sa conteneurisation avec Docker et son intégration à une extension Chrome. Il met en pratique des compétences en Machine Learning, développement logiciel, validation, versioning et déploiement Cloud.
