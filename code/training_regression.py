# File: code/training_regression.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_DATA = os.path.join(BASE_DIR, "data", "fiverr_cleaned_dl.csv")

# Chargement des données transformées
df = pd.read_csv(FILE_DATA)

# Génération des embeddings de description
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["Description"].astype(str).tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(384)])  # encodage des vecteurs denses sur 384 dimensions

# Standardisation de la variable Fiabilite
scaler = StandardScaler()
fiabilite_scaled = scaler.fit_transform(df[["Fiabilite"]])
fiabilite_df = pd.DataFrame(fiabilite_scaled, columns=["Fiabilite"])

# Construction de la matrice de features
X = pd.concat([embed_df, fiabilite_df], axis=1)
y_log = df["Prix_log"]
y_real = df["Prix"]

# Séparation en train/test
X_train, X_test, y_train, y_test, y_real_train, y_real_test = train_test_split(
    X, y_log, y_real, test_size=0.2, random_state=42
)

# Entraînement du modèle Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction et calcul du RMSE sur les vrais prix
y_pred = np.expm1(model.predict(X_test))  # inverse log transformation
rmse = np.sqrt(mean_squared_error(y_real_test, y_pred))
mae = mean_absolute_error(y_real_test, y_pred)
r2 = r2_score(y_real_test, y_pred)
print(f"MAE  : {round(mae, 2)}")
print(f"RMSE : {round(rmse, 2)}")
print(f"R²   : {round(r2, 4)}")

# Sauvegarde des modèles
os.makedirs("models/regression", exist_ok=True)
joblib.dump(model, "models/regression/gradient_boosting.pkl")
joblib.dump(scaler, "models/regression/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/columns_used.pkl")

print("Modèle et artefacts sauvegardés dans models/ et models/regression/")