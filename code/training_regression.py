# File: code/training_regression.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_DATA = os.path.join(BASE_DIR, "data", "fiverr_cleaned_dl.csv")

# Chargement des données transformées
df = pd.read_csv(FILE_DATA)

# Génération des embeddings de description
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["Description"].astype(str).tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(384)])

# Standardisation de la variable Fiabilite
scaler = StandardScaler()
fiabilite_scaled = scaler.fit_transform(df[["Fiabilite"]])
fiabilite_df = pd.DataFrame(fiabilite_scaled, columns=["Fiabilite"])

# Construction de la matrice de features
X = pd.concat([embed_df, fiabilite_df], axis=1)
y_log = df["Prix_log"]
y_real = df["Prix"]

# Séparation en train/test
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y_log, test_size=0.2, random_state=42
#)
X_train, X_test, y_train, y_test, y_real_train, y_real_test = train_test_split(
    X, y_log, y_real, test_size=0.2, random_state=42
)

# Entraînement du modèle Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction et calcul du RMSE sur les vrais prix
y_pred = np.expm1(model.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_real_test, y_pred))
#y_pred = np.expm1(model.predict(X_test))
#y_real_test = np.expm1(y_test)
#rmse = np.sqrt(mean_squared_error(y_real_test, y_pred))


# Sauvegarde du modèle
os.makedirs("models/regression", exist_ok=True)

joblib.dump(model, "models/regression/gradient_boosting.pkl")
print("Modèle sauvegardé : models/regression/gradient_boosting.pkl")

# Sauvegarde du scaler et des colonnes utilisées
joblib.dump(scaler, "models/regression/scaler.pkl")
print("scaler sauvegardé : models/regression/scaler.pkl")

joblib.dump(X.columns.tolist(), "models/columns_used.pkl")
print("colonnes sauvegardées : models/columns_used.pkl")

# Affichage final pour validation
print(f"Modèle entraîné : Gradient Boosting Regressor — RMSE = {round(rmse, 2)}")