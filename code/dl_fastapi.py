# File: code/api_fastapi.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# === Détection des chemins absolus ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "deep", "deep_model.h5"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "deep", "scaler.pkl"))

# === Initialisation de l'application ===
app = FastAPI(title="API Prédiction de Prix avec Deep Learning")

deep_model = None
scaler = None

# === Chargement du modèle d'embedding ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Mapping des niveaux de vendeur ===
niveau_mapping = [1, 2, 3]

# === Chargement des modèles ===
def load_models():
    global deep_model, scaler
    if deep_model is None:
        print("Chargement du modèle...")
        deep_model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modèle et scaler chargés.")

# === Schéma de données attendues ===
class InputData(BaseModel):
    description: str
    fiabilite: float
    niveau: int

# === Prétraitement de l'entrée ===
def preprocess_input(data: InputData):
    embedding = embedding_model.encode([data.description]).flatten()
    niveau_ohe = [1 if data.niveau == n else 0 for n in niveau_mapping]
    features = np.hstack([embedding, niveau_ohe, [data.fiabilite]])
    features_scaled = scaler.transform([features])
    return features_scaled

# === Route de prédiction ===
@app.post("/predict")
async def predict_price(input_data: InputData):
    try:
        load_models()
        X = preprocess_input(input_data)
        y_pred = await run_in_threadpool(lambda: deep_model.predict(X)[0][0])
        return {
            "prix": float(y_pred),
            #"tranche": None  # tu peux remplacer par une vraie tranche si tu veux
        }
    except Exception as e:
        return {"error": str(e)}
