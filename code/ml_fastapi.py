# File: code/api_fastapi_ml.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import numpy as np
import os

# === Répertoires et chemins ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_MODEL_PATH = os.path.join(BASE_DIR, "models/regression/gradient_boosting.pkl")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "models/classification/decision_tree.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/regression/scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models/columns_used.pkl")

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
    # Embedding
    emb = embedding_model.encode([description])
    emb_dict = {f"emb_{i}": emb[0][i] for i in range(384)}
    
    # Fiabilité pondérée
    fiabilite_pondérée = fiabilite * 0.8
    row = {**emb_dict, "Fiabilite": fiabilite_pondérée}
    df = pd.DataFrame([row])
    
    # Aligner les colonnes et scaler la fiabilité
    df = df.reindex(columns=columns, fill_value=0)
    df[["Fiabilite"]] = scaler.transform(df[["Fiabilite"]])
    
    return df

# === Route : prédiction de prix ===
@app.post("/predict_price")
def predict_price_api(data: InputData):
    X = preprocess_input(data.description, data.fiabilite)
    y_pred = reg_model.predict(X)[0]
    return {"prix": round(float(y_pred), 2)}

# === Route : prédiction de tranche ===
@app.post("/predict_tranche")
def predict_tranche_api(data: InputData):
    X = preprocess_input(data.description, data.fiabilite)
    y_pred = clf_model.predict(X)[0]
    return {"tranche": str(y_pred)}
