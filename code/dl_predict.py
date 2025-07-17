import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# 🔁 Chargement des modèles une seule fois
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
deep_model = None
scaler_dl = None

# ✅ Mapping niveau numérique
niveau_mapping = [1, 2, 3]

def load_deep_models():
    global deep_model, scaler_dl
    if deep_model is None:
        deep_model = load_model("../models/deep/deep_model.h5")
        scaler_dl = joblib.load("../models/deep/scaler.pkl")

def predict_price_dl(description, fiabilite, niveau=1):
    """
    Prédit un prix avec le modèle deep learning local (Keras).
    
    Paramètres :
    - description : str → description textuelle du service
    - fiabilite : float (entre 0 et 1)
    - niveau : int (1, 2 ou 3)

    Retourne :
    - prix : float
    """
    load_deep_models()

    emb = embedding_model.encode([description]).flatten()
    niveau_ohe = [1 if niveau == n else 0 for n in niveau_mapping]
    features = np.hstack([emb, niveau_ohe, [fiabilite]])
    features_scaled = scaler_dl.transform([features])
    prix = deep_model.predict(features_scaled)[0][0]
    return round(prix, 2)
