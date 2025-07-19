import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Chargement des modèles une seule fois
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
deep_model = None
scaler_dl = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deep", "deep_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "deep", "scaler.pkl")

# Mapping niveau numérique
niveau_mapping = [1, 2, 3]

def load_deep_models():
    global deep_model, scaler_dl
    if deep_model is None:
        deep_model = load_model(MODEL_PATH)
        scaler_dl = joblib.load(SCALER_PATH)

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
    prix = round(deep_model.predict(features_scaled, verbose=0)[0][0] * 10, 2)
    return prix
