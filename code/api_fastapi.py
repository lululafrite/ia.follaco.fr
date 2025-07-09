# File: code/api_fastapi.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool

# Création de l'application FastAPI
app = FastAPI(title="API Prédiction de Prix avec Deep Learning")

# Chargement du modèle de deep learning (Keras) et du scaler utilisé lors de l'entraînement
MODEL_PATH = "models/deep/deep_model.h5"
SCALER_PATH = "models/deep/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Chargement du modèle d'embedding pour convertir la description en vecteurs numériques
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encodage one-hot du niveau du vendeur — ordre à respecter (doit être le même qu'à l'entraînement)
niveau_mapping = ["Nouveau", "Confirmé", "Top"]

# Définition du schéma de données attendues en entrée de l'API
class InputData(BaseModel):
    Description: str
    Niveau: str
    Fiabilite: float

# Fonction de prétraitement : transforme les données utilisateur en vecteurs exploitables par le modèle
def preprocess_input(data: InputData):
    # Embedding du texte (384 dimensions)
    embedding = embedding_model.encode([data.Description]).flatten()

    # Encodage one-hot du niveau vendeur (3 dimensions)
    niveau_ohe = [1 if data.Niveau == niveau else 0 for niveau in niveau_mapping]

    # On assemble le tout dans le bon ordre : embeddings + niveau + fiabilité
    features = np.hstack([embedding, niveau_ohe, [data.Fiabilite]])

    # Application du même scaler que celui utilisé à l'entraînement
    features_scaled = scaler.transform([features])
    return features_scaled

# Route POST /predict — retourne une prédiction de prix
@app.post("/predict")
async def predict_price(input_data: InputData):
    try:
        # Prétraitement des données
        X = preprocess_input(input_data)

        # Prédiction dans un thread séparé pour éviter les plantages TensorFlow
        y_pred = await run_in_threadpool(lambda: model.predict(X)[0][0])

        return {"prix_predit": round(float(y_pred), 2)}

    except Exception as e:
        return {"error": str(e)}