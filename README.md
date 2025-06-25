# 🔮 Prédiction de prix sur Fiverr — Projet IA

Ce projet explore l’usage de l’intelligence artificielle pour **prédire le prix d’une prestation proposée sur la plateforme Fiverr**, à partir de ses caractéristiques textuelles (titre, description) et métadonnées (niveau du vendeur, évaluations, etc.).

Il combine des approches de **régression et classification supervisées** et se structure autour de **modèles machine learning interprétables**, avec un pipeline de traitement modulaire, reproductible et optimisé.

---

## 🎯 Objectif du projet

L’objectif est double :

1. **Prédire avec précision le prix réel** d’un service à partir des informations visibles sur sa fiche.
2. **Classifier automatiquement un service dans une tranche de prix** (par exemple : « moins de 10€ », « 10–30€ », etc.) pour des usages comme la segmentation marketing ou l’alerte sur des prix incohérents.

---

## 📦 Fonctionnalités clés

- 🔍 Nettoyage et normalisation des données brutes issues de Fiverr.
- 🧠 Modélisation par régression avec `XGBoostRegressor` et `HistGradientBoostingRegressor`.
- 🏷️ Classification en tranches de prix à l’aide de `RandomForestClassifier`, `HGBClassifier`, et `XGBClassifier`.
- 📈 Évaluation systématique des modèles (MAE, RMSE, R², F1-score, matrices de confusion).
- 💾 Sauvegarde organisée des modèles, features et jeux de données, prête à l’emploi pour réutilisation.

---

## ⚙️ Choix techniques

| Composant          | Choix retenu                             | Raison                                                                 |
|--------------------|-------------------------------------------|------------------------------------------------------------------------|
| Modèles ML         | XGBoost, RandomForest, HGB                | Performants, interprétables, adaptés aux petits jeux tabulaires        |
| Features textuelles| TF-IDF sur titre & description            | Méthode simple, robuste, avec bon pouvoir prédictif                   |
| Encodage caté.     | OneHotEncoder + log-count sur évaluations| Adapté aux variables discrètes et aux distributions biaisées           |
| Séparation         | `train_test_split` avec `random_state`    | Reproductibilité des résultats                                         |
| Évaluation         | `classification_report`, `ConfusionMatrixDisplay` | Comparaison facile des modèles                               |
| Organisation       | Notebooks indépendants + arborescence claire | Exécution modulaire, maintenance facilitée                         |

---

## 🗂️ Structure du projet

.
├── data/ # Données sources CSV
├── features/ # Données vectorisées (TF-IDF, log-count, sparse)
├── models/
│ ├── regression/ # Modèles et jeux de test pour la régression
│ └── classification/ # Modèles et jeux de test pour la classification
├── predicts/ # Fichiers de prédictions sauvegardés
├── 01_pre-processing.ipynb
├── 02_prepare_features.ipynb
├── 03_model_regression.ipynb
├── 04_model_classification.ipynb
└── README.md


---

## 🚀 Fonctionnement

1. **Prétraitement** (`01_pre-processing.ipynb`) : nettoyage du CSV Fiverr, sauvegarde de la version propre.
2. **Extraction de features** (`02_prepare_features.ipynb`) : vectorisation du texte, encodage des niveaux, export.
3. **Régression** (`03_model_regression.ipynb`) : entraînement des modèles, prédictions et sauvegardes.
4. **Classification** (`04_model_classification.ipynb`) : génération de tranches équilibrées, évaluation comparative des modèles de classification.

---

## 📊 Résultats

### Régression

| Modèle               | MAE    | RMSE   | R²     |
|----------------------|--------|--------|--------|
| XGBoost              | 12.78  | 48.87  | -0.01  |
| HistGradientBoosting | 17.25  | 49.19  | -0.02  |

### Classification (4 tranches équilibrées)

| Modèle                  | Accuracy | F1-score (macro) |
|-------------------------|----------|------------------|
| RandomForest (balanced) | 0.61     | 0.59             |
| HGB Classifier          | 0.67     | 0.66             |
| XGBoost Classifier      | **0.69** | **0.68**         |

---

## 🔧 Installation

```bash
git clone https://github.com/votre-compte/fiverr-pricing-ml.git
cd fiverr-pricing-ml
pip install -r requirements.txt
