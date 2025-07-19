# Prédiction de prix de service — Projet IA

Ce projet vise à prédire le prix d’une prestation freelance à partir de quelques éléments clés :

- la description textuelle du service,
- le niveau du vendeur (Nouveau, Confirmé, Top),
- la fiabilité estimée du profil (basée sur la note moyenne et le nombre d’évaluations).

Il repose sur un pipeline modulaire, combinant plusieurs techniques : traitement de texte, vectorisation par embeddings, modèles de régression, classification, deep learning, et une interface utilisateur Gradio.

---

## Sommaire

1. [Objectifs du projet](#1-objectifs)  
2. [Fonctionnalités clés](#2-fonctionnalités-clés)  
3. [Choix techniques](#3-choix-techniques)  
4. [Structure du projet](#4-structure-du-projet)  
5. [Fonctionnement général](#5-fonctionnement)  
   5.1. [Analyse exploratoire et modélisation (Notebooks)](#51-préparation-et-modélisation--notebooks)  
   5.2. [Scripts Python exécutables](#52-scripts-python--code-modulaire-et-exécutable)  
   5.3. [Jeux de données, modèles et résultats](#53-données-modèles-entraînés-et-résultats)  
   5.4. [API REST locale avec FastAPI](#54-api-rest)  
6. [Source des données](#6-source-des-données)  
7. [Résultats obtenus](#7-exemples-de-résultats)  
8. [Compétences mobilisées](#8-compétences-mobilisées)  
9. [Installation rapide](#9-installation-rapide)  
10. [Lancement de l’application (`main.py`)](#10-lancement-de-lapplication)  
11. [Remarques pédagogiques](#11-remarques-pédagogiques)  
12. [Contact](#12-contact)  
13. [Conclusion](#13-conclusion)

---

## 1. Objectifs

1. Prédire le prix exact d’un service à partir de sa description et des attributs du vendeur, en testant différentes approches (régression classique, deep learning).  
2. Classifier le service dans une **tendance de prix sur le marché** : `Basse`, `Moyenne` ou `Haute`.  
3. Proposer une interface interactive pour tester les modèles sur des cas réels.  
4. Intégrer un modèle de deep learning basé sur les embeddings pour comparer les approches.

<p align="center">
  <img src="./img/app_gradio.png" alt="Illustration de l'application Gradio" width="auto"/>
</p>

---

## 2. Fonctionnalités clés

- Nettoyage des données et suppression des outliers sur les prix.
- Création de deux versions du dataset : pour ML (`fiverr_cleaned_ml.csv`) et DL (`fiverr_cleaned_dl.csv`).
- Génération d’embeddings à partir des descriptions textuelles avec `sentence-transformers`.
- Entraînement de plusieurs types de modèles :
  - Régression : `GradientBoostingRegressor`
  - Classification : `DecisionTreeClassifier`
  - Deep learning : modèle MLP combinant embeddings + variables tabulaires
- Interface Gradio permettant de prédire un prix et une tendance de prix.
- API REST locale (FastAPI) pour interroger les modèles ML et DL.
- Log automatique des prédictions dans un fichier CSV (`log.csv`).

---

## 3. Choix techniques

| Composant            | Choix retenu                         | Raison                                                                   |
|----------------------|--------------------------------------|--------------------------------------------------------------------------|
| Texte libre          | Embeddings BERT (`all-MiniLM-L6-v2`) | Bonne représentation sémantique avec un modèle léger et rapide           |
| Variables numériques | StandardScaler                       | Améliore la convergence des modèles                                      |
| Algorithmes ML       | Gradient Boosting, Decision Tree     | Performants, rapides, explicables                                        |
| Deep learning        | MLP combiné (texte + tabulaire)      | Plus souple et adapté à des cas complexes                                |
| Interface            | Gradio                               | Permet de tester facilement les modèles                                  |
| Déploiement          | FastAPI + `main.py`                  | API REST + automatisation locale du projet                               |

---

## 4. Structure du projet

| Dossier/Fichier                       | Contenu                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `code/`                              | Scripts Python : prétraitement, entraînement, prédiction, API, UI      |
| `data/`                              | Données propres, formatées pour les modèles                            |
| `models/`                            | Modèles ML (régression/classif) et DL sauvegardés                      |
| `notebooks/`                         | Analyse, expérimentation et visualisation                              |
| `img/`                               | Graphiques et visuels de nettoyage et interface                        |
| `flagged/`                           | Fichier `log.csv` des prédictions sauvegardées                         |
| `main.py`                            | Point d’entrée principal du projet                                     |
| `requirements.txt`                   | Dépendances du projet                                                  |

---

## 5. Fonctionnement

Le projet est organisé en **notebooks** (analyse pédagogique) et **scripts Python** (exécution réelle du pipeline).

### 5.1. Préparation et modélisation — Notebooks

| Fichier                            | Rôle                                                              |
|------------------------------------|-------------------------------------------------------------------|
| `01_pre-processing.ipynb`          | Nettoyage et traitement des données                              |
| `02_model_regression.ipynb`        | Entraînement de plusieurs modèles de régression                  |
| `03_model_classification.ipynb`    | Modèles de classification des tendances de prix                  |
| `04_model_deep_learning.ipynb`     | Architecture du modèle DL et entraînement                        |
| `05_predict.ipynb`                 | Prédiction locale avec des jeux de tests                         |
| `06_api_fastapi.ipynb`             | Présentation de l’API REST                                       |
| `07_app_gradio.ipynb`              | Application Gradio interactive                                   |
| `08_predict_test.ipynb`            | Comparatif croisé ML / DL sur de multiples cas                   |

### 5.2. Scripts Python — Code modulaire et exécutable

- `preprocessing.py` : nettoyage, suppression d’outliers, encodage, sauvegarde CSV  
- `training_regression.py` : modèle Gradient Boosting  
- `training_classification.py` : modèle Decision Tree  
- `training_deep.py` : entraînement du modèle deep MLP  
- `ml_predict.py`, `dl_predict.py` : fonctions de prédiction  
- `ml_fastapi.py`, `dl_fastapi.py` : APIs REST ML et DL  
- `app_gradio.py` : application interactive Gradio  
- `train_all.py` : exécute les trois entraînements d’un coup  
- `main.py` : exécute l’ensemble pipeline (vérifie, entraîne, lance API + UI)

### 5.3. Données, modèles entraînés et résultats

- `data/fiverr_cleaned_ml.csv`, `data/fiverr_cleaned_dl.csv` : jeux de données finaux  
- `models/` : modèles sauvegardés (`.pkl`, `.h5`)  
- `img/` : images utilisées dans les notebooks et ce README  
- `flagged/log.csv` : prédictions sauvegardées par l’utilisateur Gradio

### 5.4. API REST

- `ml_fastapi.py` : expose les modèles ML pour la prédiction prix + tendance  
- `dl_fastapi.py` : expose le modèle DL pour la prédiction prix uniquement

Exécution locale :
```bash
uvicorn code.ml_fastapi:app --reload
uvicorn code.dl_fastapi:app --reload
```

Cette structure permet de séparer clairement l’exploration (notebooks), le code exécutable (scripts), les données, les modèles et les résultats.

---

## 6. Source des données

Jeu de données Kaggle :  
https://www.kaggle.com/datasets/muhammadadiltalay/fiverr-data-gigs

Le fichier `fiverr-data-gigs.csv` a été nettoyé et transformé localement.  
Deux fichiers sont produits :
- `fiverr_cleaned_ml.csv` : pour les modèles classiques (ML)
- `fiverr_cleaned_dl.csv` : avec transformation `log` du prix pour le modèle DL

---

## 7. Exemples de résultats

### 7.1 Régression (ML)

| Modèle                 | MAE   | RMSE   | R²         |
|------------------------|-------|--------|------------|
| **Gradient Boosting**  | 3.21  | 4.90   | **0.2566** |

### 7.2 Classification (ML)

| Modèle               | Accuracy   |
|----------------------|------------|
| **Decision Tree**    | **61.75 %** |

### 7.3 Deep Learning

| Modèle                | MAE   | RMSE   | R²         |
|-----------------------|-------|--------|------------|
| **MLP (DL)**          | 3.17  | 4.85   | 0.2710     |

---

## 8. Compétences mobilisées

| Bloc   | Compétence | Description                                                                                          |
|--------|------------|------------------------------------------------------------------------------------------------------|
| Bloc_3 | C1         | Sélection des algorithmes de régression et classification                                            |
| Bloc_3 | C2         | Nettoyage, encodage, standardisation, embeddings textuels                                            |
| Bloc_3 | C3         | Entraînement ML avec GridSearchCV et validation croisée                                              |
| Bloc_5 | C1         | Transformation texte en vecteurs via `sentence-transformers`                                         |
| Bloc_5 | C2         | Modèle hybride texte + variables numériques                                                          |
| Bloc_5 | C3         | Entraînement MLP avec EarlyStopping, séparation validation                                           |
| Bloc_5 | C4         | Déploiement API (FastAPI) + Application utilisateur (Gradio)                                         |

---

## 9. Installation rapide

```bash
git clone https://github.com/lululafrite/ia.follaco.fr.git
cd ia.follaco.fr
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 10. Lancement de l'application

Le fichier `main.py` constitue le **point d’entrée principal** pour tester l’ensemble du projet.

Il permet :

1. de vérifier la présence des modèles (régression, classification, deep learning),
2. d’entraîner automatiquement les modèles si besoin (`train_all.py`),
3. de lancer l’API REST avec FastAPI en arrière-plan,
4. de démarrer l’application Gradio pour tester les prédictions.

### 10.1. Exécution

Dans un terminal (à la racine du projet), lancez simplement :

```bash
python main.py
```
---

## 11. Remarques pédagogiques

- Chaque notebook est autonome, avec une structure claire et des explications guidées.
- Le projet est conçu pour favoriser la modularité et la réutilisabilité en production.
- L’interface utilisateur permet de tester directement toute la chaîne de prédiction.
- Les notebooks `05_predict.ipynb` et `06_api_fastapi.ipynb` ne sont pas utilisés par l’application. Ils ont un rôle purement pédagogique.
- Le notebook `08_predict_test.ipynb` n’est pas utilisé par l’application. Il a un rôle analytique et pédagogique.
- Le fichier `ml_predict.py` et `dl_predict.py` sont importés dans le notebook `07_app_gradio.ipynb`, car il contient les fonctions de prédiction.

---

## 12. Contact

Ludovic FOLLACO  
ludovic.follaco@gmail.com  
https://www.follaco.fr  
https://www.linkedin.com/in/ludovic-follaco-a74b5394/

---

## 13. Conclusion

Ce projet démontre ma capacité à concevoir un pipeline IA complet :

depuis la collecte et la préparation des données,

jusqu’au déploiement d’une interface interactive.

Il compare les approches classiques (ML) et avancées (DL), avec un focus sur la modularité, l’évolutivité, et l’intégration dans un cas d’usage réel (freelance pricing).