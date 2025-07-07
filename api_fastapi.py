from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# 🌐 Création de l'application FastAPI
app = FastAPI(title="API Prédiction de Prix avec Deep Learning")

# 📦 Chargement du modèle Keras et du scaler
MODEL_PATH = "models/deep/deep_model.h5"
SCALER_PATH = "models/deep/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 🔤 Chargement du modèle d'embedding pour la description
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🎯 Mapping pour l'encodage one-hot du niveau
niveau_mapping = ["Nouveau", "Confirmé", "Top"]

# 🧾 Définition du schéma d'entrée avec Pydantic
class InputData(BaseModel):
    Description: str
    Niveau: str
    Fiabilite: float

# 🔍 Fonction de prétraitement
def preprocess_input(data: InputData):
    # Embedding de la description
    #embedding = embedding_model.encode([data.Description])
    embedding = embedding_model.encode([data.Description]).flatten()

    # One-hot encoding du niveau
    niveau_ohe = [1 if data.Niveau == niveau else 0 for niveau in niveau_mapping]

    # Construction du tableau final
    features = np.hstack([embedding, niveau_ohe, [data.Fiabilite]])
    features_scaled = scaler.transform([features])
    return features_scaled

# 📨 Route de prédiction
@app.post("/predict")
def predict_price(input_data: InputData):
    try:
        X = preprocess_input(input_data)
        y_pred = model.predict(X)[0][0]
        return {"prix_predit": round(float(y_pred), 2)}
    except Exception as e:
        return {"error": str(e)}
