# Prédiction de prix de service — Projet IA

Ce projet vise à prédire le prix d’une prestation freelance à partir de quelques éléments clés :

- la description textuelle du service,
- le niveau du vendeur (Nouveau, Confirmé, Top),
- la fiabilité estimée du profil (basée sur la note moyenne et le nombre d’évaluations).

Il repose sur un pipeline modulaire, combinant plusieurs techniques : traitement de texte, vectorisation par embeddings, modèles de régression, classification, et une interface utilisateur développée avec Gradio.

---

## Sommaire

1. [Objectifs du projet](#1-objectifs)  
2. [Fonctionnalités clés](#2-fonctionnalités-clés)  
3. [Choix techniques](#3-choix-techniques)  
4. [Structure du projet](#4-structure-du-projet)  
5. [Fonctionnement général](#5-fonctionnement)  
   5.1. [Analyse exploratoire et modélisation (Notebooks)](#51-préparation-et-modélisation--notebooks)  
   5.2. [Scripts Python modulaires](#52-scripts-python--code-modulaire-et-exécutable)  
   5.3. [Jeux de données, modèles et prédictions](#53-données-modèles-entraînés-et-résultats)  
   5.4. [API REST locale avec FastAPI](#54-api-rest)  
6. [Source des données](#6-source-des-données)  
   6.1. [Utilisation des fichiers intermédiaires](#61-note-sur-les-fichiers-intermédiaires)  
7. [Résultats obtenus](#7-exemples-de-résultats)  
   7.1. [Modèles de régression](#71-régression--comparatif-des-modèles)  
   7.2. [Modèles de classification](#72-classification--comparatif-des-modèles)  
   7.3. [Approche hybride : Deep Learning](#73-approche-hybride--deep-learning-vs-modèles-classiques)  
8. [Compétences mobilisées](#8-compétences-mobilisées)  
9. [Installation rapide](#9-installation-rapide)  
10. [Lancement de l’application (`main.py`)](#10-lancement-de-lapplication)  
   10.1. [Exécution](#101-exécution)  
11. [Remarques pédagogiques](#11-remarques-pédagogiques)  
12. [Contact](#12-contact)  
13. [Conclusion](#13-conclusion)

---

## 1. Objectifs

1. Prédire le prix exact d’un service à partir de sa description et des attributs du vendeur, en testant différentes approches (régression classique, deep learning).
2. Classifier le service dans une tranche de prix : `Basse`, `Moyenne` ou `Haute`.  
3. Proposer une interface interactive pour tester les modèles sur des cas réels.
4. Intégrer un modèle de deep learning basé sur les embeddings pour comparer les approches.

<p align="center">
  <img src="./img/app_gradio.png" alt="Illustration de l'application Gradio" width="auto"/>
</p>

---

## 2. Fonctionnalités clés

- Nettoyage complet des données brutes issues de Fiverr.
- Génération d’embeddings à partir des descriptions textuelles avec `sentence-transformers`.
- Entraînement de plusieurs types de modèles :
  - Régression : `GradientBoostingRegressor`
  - Classification : `DecisionTreeClassifier`
  - Deep learning : modèle combinant embeddings et variables numériques
- Évaluation des modèles avec les métriques standards : `MAE`, `RMSE`, `Accuracy`, `F1-score`.
- Sauvegarde des modèles, des colonnes utilisées, des transformateurs et des prédictions.
- Application interactive développée avec Gradio pour tester les différents modèles.
- API REST disponible pour accéder aux prédictions via une requête HTTP.

---

## 3. Choix techniques

| Composant            | Choix retenu                         | Raison                                                                   |
|----------------------|--------------------------------------|--------------------------------------------------------------------------|
| Texte libre          | Embeddings BERT (`all-MiniLM-L6-v2`) | Bonne représentation sémantique avec un modèle léger et rapide           |
| Variables numériques | StandardScaler                       | Améliore la convergence des modèles et les rend plus stables             |
| Algorithmes ML       | Gradient Boosting, Decision Tree     | Bons résultats, interprétables, adaptés à des volumes de données limités |
| Deep learning        | MLP combinant embeddings + numériques| Permet de comparer une approche dense à la régression classique          |
| Interface            | Gradio                               | Permet de tester facilement l’ensemble des modèles                       |
| Déploiement          | API FastAPI                          | Exposition d’un modèle de prédiction via une API REST locale             |
| Organisation         | Séparation notebooks / code          | Structure claire entre analyse et exécution                              |

---

## 4. Structure du projet

| Dossier                               | Contenu du dossier                                                         |
|---------------------------------------|----------------------------------------------------------------------------|
| ia.follaco.fr/                        | Dossier racine du projet                                                   |
| ia.follaco.fr/data/                   | Données brutes et fichiers transformés (CSV)                               |
| ia.follaco.fr/data/predictions_grid/  | Résultats de prédictions croisant les modèles de régression/classification |
| ia.follaco.fr/flagged/                | Contient le fichier `log.csv` utilisé pour enregistrer les prédictions     |
| ia.follaco.fr/img/                    | Images utilisées dans les notebooks et le README                           |
| ia.follaco.fr/models/                 | Modèles entraînés (régression, classification, deep learning)              |
| ia.follaco.fr/models/classification/  | Modèles de classification enregistrés                                      |
| ia.follaco.fr/models/regression/      | Modèles de régression enregistrés                                          |
| ia.follaco.fr/models/deep/            | Modèles de deep learning enregistrés                                       |
| ia.follaco.fr/notebooks/              | Notebooks d’analyse et de démonstration (prétraitement, modèles, etc.)     |
| ia.follaco.fr/code/                   | Scripts Python exécutables : fonctions, API REST, application Gradio       |


---

## 5. Fonctionnement

Le projet est organisé en plusieurs grands ensembles, chacun ayant un rôle bien défini.  
Les notebooks servent à documenter chaque étape du projet, tandis que les fichiers Python permettent d'exécuter le code plus simplement, de manière modulaire et réutilisable.

### 5.1. Préparation et modélisation — Notebooks

Ces fichiers servent à explorer les données, tester plusieurs approches, visualiser les résultats et conserver une trace claire de chaque étape. Ils sont regroupés dans le dossier `notebooks/`.

- `01_pre-processing.ipynb` : nettoyage des données brutes  
- `02_model_regression.ipynb` : entraînement et comparaison de plusieurs modèles de régression  
- `03_model_classification.ipynb` : entraînement et comparaison de modèles de classification  
- `04_model_deep_learning.ipynb` : mise en place d’un modèle de deep learning basé sur les embeddings  
- `05_predict.ipynb` : test de la prédiction sur de nouvelles entrées (local)  
- `06_api_fastapi.ipynb` : présentation de l’API REST  
- `07_app_gradio.ipynb` : test de l’application Gradio  
- `08_predict_test.ipynb` : comparaison des paires de modèles "régression/classification" pour analyse

Ces notebooks permettent aussi de justifier les choix de modèles.


### 5.2. Scripts Python — Code modulaire et exécutable

Tous les scripts utiles au fonctionnement réel du projet sont regroupés dans le dossier `code/`. Ils sont organisés par fonctionnalité.

- **Prétraitement et préparation des données**  
  - `preprocessing.py` : fonctions de nettoyage des données brutes  
  - `features.py` : encodage, embeddings, standardisation

- **Entraînement des modèles**  
  - `training_regression.py` : régression supervisée  
  - `training_classification.py` : classification des tranches de prix  
  - `training_deep.py` : deep learning basé sur embeddings

- **Prédiction**  
  - `predict.py` : contient toutes les fonctions utilisées pour prédire un prix ou une tranche à partir d'une nouvelle entrée

- **Déploiement**  
  - `api_fastapi.py` : API REST basée sur FastAPI  
  - `app_gradio.py` : interface utilisateur Gradio pour tester les modèles  
  - `main.py` : point d’entrée central si on veut enchaîner plusieurs étapes

### 5.3. Données, modèles entraînés et résultats

- `data/` : contient le dataset brut et les fichiers transformés (`.csv`)
- `data/predictions_grid/` : résultats des prédictions croisant plusieurs modèles de régression et classification
- `models/` : tous les modèles sauvegardés (.pkl ou .h5), organisés par type (régression, classification, deep)
- `flagged/log.csv` : journal des prédictions faites par l’utilisateur via Gradio
- `img/` : visuels utilisés pour la soutenance ou dans les notebooks

### 5.4. API REST

Le fichier `api_fastapi.py` permet de lancer une API REST en local.  
Elle est construite avec le framework FastAPI, connu pour sa légèreté et sa rapidité.  
L’API reçoit une description, un niveau de vendeur et une fiabilité, puis retourne un prix prédit.

Elle est utilisée dans certains notebooks ou depuis l’interface Gradio pour simuler une interaction avec un service externe.

Le fichier `api_fastapi.py` n’est pas exécuté depuis le notebook.  
Il doit être lancé à part, dans un terminal, avec la commande suivante :

```bash
uvicorn code.api_fastapi:app --reload
```

Cette structure permet de séparer clairement l’exploration (notebooks), le code exécutable (scripts), les données, les modèles et les résultats.

---

## 6. Source des données

Les données utilisées pour entraîner les modèles viennent d’un csv disponible publiquement sur Kaggle.  
Il s’agit d’une extraction de prestations proposées sur la plateforme Fiverr : description, prix, niveau du vendeur, nombre d’avis, etc.  
Le fichier original a été nettoyé et retravaillé localement pour ne garder que les variables utiles au projet.

Lien Kaggle : https://www.kaggle.com/datasets/muhammadadiltalay/fiverr-data-gigs

### 6.1. Note sur les fichiers intermédiaires

Le projet conserve uniquement les fichiers essentiels au fonctionnement de l’application, notamment `fiverr_cleaned_transformed.csv` utilisé pour l’inférence.

Les autres fichiers CSV présents dans les dossiers `data/predictions_grid/` et `flagged/` sont conservés à des fins pédagogiques :

- comparaison des modèles de régression et classification,
- journalisation des prédictions effectuées via l’interface Gradio.

Ces fichiers ne sont pas requis pour le bon déroulement du pipeline de production.

---

## 7. Exemples de résultats

Cette section présente les résultats obtenus lors de l'entraînement et de l’évaluation des différents modèles.  
Elle permet de comparer objectivement les performances et de justifier les choix retenus pour la suite du projet.

---

### 7.1 Régression — Comparatif des modèles

| Modèle                 | MAE   | RMSE   | R²         |
|------------------------|-------|--------|------------|
| **Gradient Boosting**  | 3.21  | 4.90   | **0.2566** |
| XGBoost                | 3.33  | 5.00   | 0.2274     |
| Random Forest          | 3.32  | 5.03   | 0.2173     |
| Ridge                  | 3.82  | 5.47   | 0.0748     |
| KNN Regressor          | 3.99  | 5.80   | -0.0407    |
| Decision Tree          | 4.43  | 7.14   | -0.5770    |
| Linear Regression      | 5.86  | 9.90   | -2.0353    |

**Modèle retenu** : `Gradient Boosting Regressor` — meilleur compromis entre précision et stabilité (RMSE = 4.90).

---

### 7.2 Classification — Comparatif des modèles

| Modèle               | Accuracy   |
|----------------------|------------|
| **Decision Tree**    | **0.6175** |
| Random Forest        | 0.5339     |
| KNN Classifier       | 0.5179     |
| Logistic Regression  | 0.5100     |

**Modèle retenu** : `Decision Tree Classifier` — meilleur score d’accuracy (61.75 %) pour classer les tranches Basse / Moyenne / Haute.

> Ces résultats ont motivé la mise en place d’un modèle plus avancé, combinant texte et variables numériques.

---

### 7.3 Approche hybride — Deep learning vs modèles classiques

Le modèle de deep learning a été entraîné à partir d’un vecteur combinant :
- les embeddings textuels de la description (`sentence-transformers`)
- les variables numériques standardisées (fiabilité, niveau de vendeur encodé)

Il est comparé ici au meilleur modèle classique de régression (`Gradient Boosting Regressor`).  
Les deux modèles ont été évalués sur le même jeu de validation pour garantir une comparaison équitable.

| Modèle                       | MAE   | RMSE  | R²       |
|------------------------------|-------|-------|----------|
| Gradient Boosting Regressor  | 3.21  | 4.90  | 0.2566   |
| Deep learning (MLP)          | 3.17  | 4.85  | 0.2710   |

**Conclusion** : Le modèle de deep learning offre un léger gain en précision, tout en conservant une architecture simple. Il permet de valider l’intérêt d’une approche hybride dans le cadre de ce projet.

---

## 8. Compétences mobilisées

Ce projet mobilise des compétences issues des blocs 3 et 5 de la certification IA.  
Chaque compétence est couverte à travers des notebooks d’analyse et des scripts Python exécutables.

| Bloc   | Compétence | Description                                                                                          | Fichiers associés                                                                                     |
|--------|------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Bloc_3 | C1         | Sélection du meilleur algorithme ML selon les performances (MAE, RMSE, Accuracy…)                   | `02_model_regression.ipynb`, `03_model_classification.ipynb`, `training_regression.py`, `training_classification.py` |
| Bloc_3 | C2         | Prétraitement des données (scaling, encodage, embeddings) adapté à chaque type de modèle            | `01_pre-processing.ipynb`, `features.py`, `preprocessing.py`                                          |
| Bloc_3 | C3         | Entraînement et validation de modèles ML supervisés sur métriques définies                          | `02_model_regression.ipynb`, `03_model_classification.ipynb`, `training_regression.py`, `training_classification.py` |
| Bloc_5 | C1         | Transformation de textes en vecteurs numériques (embeddings BERT via sentence-transformers)         | `features.py`, `04_model_deep_learning.ipynb`, `training_deep.py`                                     |
| Bloc_5 | C2         | Comparaison de modèles hybrides (ML vs Deep Learning) adaptés aux contraintes du cas métier         | `04_model_deep_learning.ipynb`, `02_model_regression.ipynb`, `08_predict_test.ipynb`                  |
| Bloc_5 | C3         | Entraînement de modèles Deep Learning exploitant les embeddings textuels                            | `04_model_deep_learning.ipynb`, `training_deep.py`                                                    |
| Bloc_5 | C4         | Déploiement du pipeline avec une interface Gradio et exposition d’un modèle via une API FastAPI     | `app_gradio.py`, `api_fastapi.py`, `06_app_gradio.ipynb`, `06_api_fastapi.ipynb`, `main.py`                  |

---

## 9. Installation rapide

**Prérequis**  
Le projet a été développé et testé avec :
- Python 3.10.11
- Les packages listés dans le fichier `requirements.txt`

### 9.1. Étapes d'installation

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
- Les notebooks `04_predict.ipynb` et `06_api_fastapi.ipynb` ne sont pas utilisés par l’application. Ils ont un rôle purement pédagogique.
- Le notebook `08_predict_test.ipynb` n’est pas utilisé par l’application. Il a un rôle analytique et pédagogique.
- Le fichier `predict.py` est importé dans le notebook `07_app_gradio.ipynb`, car il contient les fonctions de prédiction.

---

## 12. Contact

Ludovic FOLLACO  
ludovic.follaco@gmail.com  
https://www.follaco.fr  
https://www.linkedin.com/in/ludovic-follaco-a74b5394/

---

## 13. Conclusion

Ce projet a permis d'explorer différentes approches pour prédire le prix d’un service en ligne à partir de données textuelles et numériques.  
En combinant des techniques classiques (régression, classification) avec du deep learning, il illustre la complémentarité des méthodes et l’intérêt de tester plusieurs modèles pour répondre à une problématique métier.

La structure choisie permet de séparer clairement l’analyse, l’entraînement, la prédiction et le déploiement.  
L’ensemble du code est modulaire, reproductible et facilement testable via l’interface Gradio ou l’API REST.

Ce travail peut servir de base pour d’autres projets de prédiction de valeur sur des plateformes similaires.