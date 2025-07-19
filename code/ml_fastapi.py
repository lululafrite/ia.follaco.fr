# File: code/api_fastapi_ml.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import numpy as np
import os

# Définition du répertoire de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Définition des chemins vers les fichiers nécessaires
REG_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "regression", "gradient_boosting.pkl"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "scaler.pkl"))
CLF_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "decision_tree.pkl"))
COLUMNS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "classification", "columns_used.pkl"))

# === Chargement des modèles ===
reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Application FastAPI ===
app = FastAPI(title="API ML - Prédiction de prix et de tranche")

# === Schéma d’entrée ===
class InputData(BaseModel):
    description: str
    fiabilite: float

# === Prétraitement commun ===
def preprocess_input(description: str, fiabilite: float) -> pd.DataFrame:
    emb = embedding_model.encode([description])
    emb_dict = {f"emb_{i}": emb[0][i] for i in range(384)}

    # Ne pas pondérer ici
    row = {**emb_dict, "Fiabilite": fiabilite}
    df = pd.DataFrame([row])

    df = df.reindex(columns=columns, fill_value=0)
    df[["Fiabilite"]] = scaler.transform(df[["Fiabilite"]])
    return df

# === Route : prédiction de prix ===
@app.post("/predict_price")
def predict_price_api(data: InputData):
    # On applique la pondération ici, comme dans ml_predict.py
    fiabilite_pondérée = data.fiabilite * 0.8
    X = preprocess_input(data.description, fiabilite_pondérée)
    y_pred = reg_model.predict(X)[0]
    return {"prix": round(float(y_pred) * 10, 2)}

# === Route : prédiction de tranche ===
@app.post("/predict_tranche")
def predict_tranche_api(data: InputData):
    fiabilite_pondérée = data.fiabilite * 0.8
    X = preprocess_input(data.description, fiabilite_pondérée)
    y_pred = clf_model.predict(X)[0]
    return {"tranche": str(y_pred)}