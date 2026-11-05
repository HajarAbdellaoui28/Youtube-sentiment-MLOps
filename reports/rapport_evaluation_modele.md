# Rapport d'évaluation du modèle de classification des sentiments

**Projet :** Analyse de sentiment des commentaires YouTube — MLOps  
**Source des résultats :** exécution de `python src/models/train.py`  
**Modèle retenu :** `LinearSVC` associé à une vectorisation `TF-IDF`  
**Jeu évalué :** 7 313 observations

## 1. Objectif

Ce rapport présente les performances du modèle de classification des sentiments entraîné dans le cadre du projet MLOps. La tâche consiste à prédire l'une des trois classes : **négatif (-1)**, **neutre (0)** et **positif (1)**.

## 2. Sélection du modèle et hyperparamètres

L'entraînement a utilisé une recherche par validation croisée sur **3 folds**, avec **48 configurations candidates**, soit **144 ajustements (fits)** affichés par le script.

| Paramètre | Valeur retenue |
|---|---|
| Classificateur | `LinearSVC()` |
| `classifier__C` | `1` |
| `classifier__class_weight` | `balanced` |
| `tfidf__max_features` | `10000` |
| `tfidf__ngram_range` | `(1, 1)` |

La configuration utilise donc une représentation TF-IDF limitée à 10 000 caractéristiques, avec des unigrammes. Le poids des classes est équilibré dans le classificateur.

## 3. Résultats globaux

| Métrique | Résultat |
|---|---:|
| Accuracy (valeur brute affichée) | `0.8993573089019554` |
| Accuracy (pourcentage) | **89,94 %** |
| Precision macro | **0,89** |
| Recall macro | **0,89** |
| F1-score macro | **0,89** |
| Precision pondérée | **0,90** |
| Recall pondéré | **0,90** |
| F1-score pondéré | **0,90** |
| Taille de l'ensemble évalué | **7 313** |


## 4. Rapport de classification par classe

| Classe | Sentiment | Precision | Recall | F1-score | Support |
|---|---|---:|---:|---:|---:|
| -1 | Négatif | 0,86 | 0,80 | 0,83 | 1 647 |
| 0 | Neutre | 0,90 | 0,97 | 0,93 | 2 521 |
| 1 | Positif | 0,92 | 0,90 | 0,91 | 3 145 |
| | **Macro avg** | **0,89** | **0,89** | **0,89** | **7 313** |
| | **Weighted avg** | **0,90** | **0,90** | **0,90** | **7 313** |

### Interprétation

- **Classe neutre :** meilleur F1-score (0,93) et meilleur rappel (0,97). Le modèle reconnaît correctement la plupart des exemples neutres.
- **Classe positive :** F1-score de 0,91 et précision de 0,92 ; les prédictions positives sont globalement fiables sur cet ensemble.
- **Classe négative :** F1-score de 0,83 et rappel de 0,80. C'est la classe la plus difficile à détecter parmi les trois.

## 5. Matrice de confusion

**Lignes : classes réelles ; colonnes : classes prédites.**

| Réel \ Prédit | Négatif (-1) | Neutre (0) | Positif (1) | Total réel |
|---|---:|---:|---:|---:|
| **Négatif (-1)** | **1 313** | 119 | 215 | 1 647 |
| **Neutre (0)** | 35 | **2 446** | 40 | 2 521 |
| **Positif (1)** | 171 | 156 | **2 818** | 3 145 |
| **Total prédit** | 1 519 | 2 721 | 3 073 | **7 313** |

La diagonale représente les classifications correctes. La matrice indique notamment :

- **215** commentaires réellement négatifs prédits comme positifs ;
- **171** commentaires réellement positifs prédits comme négatifs ;
- **119** commentaires négatifs prédits comme neutres ;
- **156** commentaires positifs prédits comme neutres.

Ces confusions suggèrent que la distinction des polarités, en particulier pour la classe négative, reste un axe d'amélioration.


## 6. Conclusion

Le pipeline **TF-IDF + LinearSVC** atteint une **accuracy de 89,94 %** sur **7 313 observations** et un **F1-score macro de 0,89** (arrondi). Les performances sont particulièrement élevées sur la classe neutre, tandis que la classe négative présente une marge d'amélioration. Ces résultats constituent une base quantitative pour documenter le modèle utilisé par l'API de prédiction du projet.

---

**Traçabilité :** valeurs retranscrites de la sortie terminal de `python src/models/train.py` communiquée pour le projet.
