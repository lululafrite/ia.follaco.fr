# Code source du projet

Ce dossier contient tous les scripts Python utilisés pour entraîner les modèles, effectuer des prédictions, lancer l’API ou l’application Gradio.

Avant de consulter ce dossier, il est recommandé de lire le fichier `README.md` situé à la racine du projet.  
Il présente les objectifs du projet, la structure générale, les résultats obtenus et les compétences mobilisées.

Chaque script est indépendant, mais suit une structure modulaire cohérente avec l’ensemble du projet.

---

## Organisation des fichiers

| Fichier                      | Rôle                                                                 |
|------------------------------|----------------------------------------------------------------------|
| `preprocessing.py`           | Nettoyage des données brutes                                         |
| `features.py`                | Transformation des données : encodage, embeddings, standardisation   |
| `training_regression.py`     | Entraînement des modèles de régression                               |
| `training_classification.py` | Entraînement des modèles de classification                           |
| `training_deep.py`           | Entraînement d’un modèle de deep learning hybride                    |
| `predict.py`                 | Fonctions de prédiction appelées dans l’application Gradio           |
| `api_fastapi.py`             | API REST permettant d’exposer le modèle de prédiction                |
| `app_gradio.py`              | Application Gradio (interface utilisateur)                           |
| `main.py`                    | Point d’entrée pour enchaîner plusieurs étapes si besoin             |

---

## Exécution

- Pour tester l’interface Gradio :  
```bash
    python main.py
```
- Pour lancer l’API REST (FastAPI) :  
```bash
    uvicorn code.api_fastapi:app --reload
```