# File: code/predict.py

import os
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Définition du répertoire de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Définition des chemins vers les fichiers nécessaires
REG_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "regression", "gradient_boosting.pkl"))
SCALER_CLASS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "scaler.pkl"))
CLF_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "decision_tree.pkl"))
COLUMNS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "columns_used.pkl"))

# Chargement des objets
reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)
scaler_class = joblib.load(SCALER_CLASS_PATH)
columns = joblib.load(COLUMNS_PATH)

# Chargement du modèle d'embedding (même que celui utilisé à l'entraînement)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Fonction de prétraitement : transforme l'entrée utilisateur en vecteurs alignés avec les colonnes attendues
def preprocess_input(description: str, fiabilite: float) -> pd.DataFrame:
    # Encodage de la description en 384 dimensions
    emb = embedding_model.encode([description])
    emb_dict = {f"emb_{i}": emb[0][i] for i in range(384)}

    # Construction de la ligne complète avec la fiabilité
    row = {**emb_dict, "Fiabilite": fiabilite}
    df = pd.DataFrame([row])

    # On s'assure que toutes les colonnes attendues sont là, dans le bon ordre
    df = df.reindex(columns=columns, fill_value=0)

    # Standardisation de la fiabilité
    df[["Fiabilite"]] = scaler_class.transform(df[["Fiabilite"]])

    return df

# Prédiction de prix (régression)
def predict_price(description: str, fiabilite: float) -> float:
    # On applique une pondération arbitraire sur la fiabilité pour améliorer la cohérence
    fiabilite_pondérée = fiabilite * 0.8
    X = preprocess_input(description, fiabilite_pondérée)
    # Multiplication par 2.5 pour retrouver l’échelle initiale des prix
    return round(reg_model.predict(X)[0] * 10, 2)

# Prédiction de tranche de prix (classification)
def predict_tranche(description: str, fiabilite: float) -> str:
    # Même logique que ci-dessus, avec une pondération légèrement différente
    fiabilite_pondérée = fiabilite * 0.8
    X = preprocess_input(description, fiabilite_pondérée)
    return str(clf_model.predict(X)[0])