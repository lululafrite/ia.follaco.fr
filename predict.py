# 📦 predict.py — Pipeline cohérent avec modèle entraîné

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# Chargement des objets
reg_model = joblib.load("models/regression/gradient_boosting.pkl")
clf_model = joblib.load("models/classification/decision_tree.pkl")
scaler = joblib.load("models/regression/scaler.pkl")
columns = joblib.load("models/columns_used.pkl")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔧 Prétraitement (aligné au pipeline d'entraînement)
def preprocess_input(description: str, fiabilite: float) -> pd.DataFrame:
    emb = embedding_model.encode([description])
    emb_dict = {f"emb_{i}": emb[0][i] for i in range(384)}
    
    row = {**emb_dict, "Fiabilite": fiabilite}
    df = pd.DataFrame([row])
    df = df.reindex(columns=columns, fill_value=0)
    df[["Fiabilite"]] = scaler.transform(df[["Fiabilite"]])
    return df

# 🔮 Prédiction
def predict_price(description: str, fiabilite: float) -> float:
    # Pondération comme dans preprocessing
    # coeffs = {"Beginner": 0.8, "Intermediate": 1.0, "Expert": 1.2}
    fiabilite_pondérée = fiabilite * 0.8 # coeffs.get(niveau, 1.0)
    X = preprocess_input(description, fiabilite_pondérée)
    return round(reg_model.predict(X)[0] * 10, 2)

def predict_tranche(description: str, fiabilite: float) -> str:
    # coeffs = {"Beginner": 0.8, "Intermediate": 1.0, "Expert": 1.2}
    fiabilite_pondérée = fiabilite * 0.8 # coeffs.get(niveau, 1.0)
    X = preprocess_input(description, fiabilite_pondérée)
    return str(clf_model.predict(X)[0])