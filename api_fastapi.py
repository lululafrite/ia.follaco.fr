from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool

# 🌐 Création de l'application FastAPI
app = FastAPI(title="API Prédiction de Prix avec Deep Learning")

# 📦 Chargement du modèle Keras et du scaler utilisé lors de l'entraînement
MODEL_PATH = "models/deep/deep_model.h5"
SCALER_PATH = "models/deep/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 🔤 Chargement du modèle d'embedding pour la description
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔁 Mapping texte → entier pour le niveau vendeur
# 📌 Doit correspondre aux valeurs présentes dans le CSV d'entraînement
niveau_to_int = {
    "Nouveau": 1,
    "Confirmé": 2,
    "Top": 3
}

# 🧾 Schéma d'entrée de la requête POST via Pydantic
class InputData(BaseModel):
    Description: str
    Niveau: str
    Fiabilite: float

# 🔍 Fonction de prétraitement : transforme les données utilisateur en entrée modèle
def preprocess_input(data: InputData):
    # Embedding de la description
    embedding = embedding_model.encode([data.Description]).flatten()

    # One-hot encoding du niveau (3 colonnes)
    niveau_mapping = ["Nouveau", "Confirmé", "Top"]
    niveau_ohe = [1 if data.Niveau == niveau else 0 for niveau in niveau_mapping]

    # Fusion finale (384 + 3 + 1 = 388 colonnes)
    features = np.hstack([embedding, niveau_ohe, [data.Fiabilite]])

    print(">> Shape des features brutes :", features.shape)  # Doit être (388,)
    
    features_scaled = scaler.transform([features])
    return features_scaled


# 📨 Route principale de prédiction
@app.post("/predict")
async def predict_price(input_data: InputData):
    try:
        X = preprocess_input(input_data)

        print(">> Prédiction sur X :", X.shape)
        import sys; sys.stdout.flush()

        # ⚠️ Appel du modèle dans un thread isolé pour éviter les crashs TensorFlow
        y_pred = await run_in_threadpool(lambda: model.predict(X)[0][0])

        return {"prix_predit": round(float(y_pred), 2)}
    except Exception as e:
        import traceback
        print("❌ ERREUR PREDICTION :")
        traceback.print_exc()
        return {"error": str(e)}

