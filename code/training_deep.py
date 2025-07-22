# File: code/training_deep.py

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Chargement des données préparées
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_DATA = os.path.join(BASE_DIR, "data", "fiverr_cleaned_dl.csv")

# Chargement des données transformées
df = pd.read_csv(FILE_DATA)

# Embedding des descriptions
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["Description"].astype(str).tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

# Encodage one-hot du niveau du vendeur
niveau_encoded = pd.get_dummies(df["Niveau"], prefix="Niveau") # tableau d'affectation binaire

# Ajout de la variable Fiabilite
autres_features = df[["Fiabilite"]].reset_index(drop=True)

# Fusion des features
X = pd.concat([embed_df, niveau_encoded, autres_features], axis=1)
y_log = df["Prix_log"]

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde du scaler
os.makedirs("models/deep", exist_ok=True)
joblib.dump(scaler, "models/deep/scaler.pkl")

# Définition du modèle MLP (régression)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entraînement avec EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Évaluation finale sur le jeu de test
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Évaluation finale - MAE : {mae:.2f}, MSE : {loss:.2f}")

# === Évaluation finale en euros réels ===
y_pred = model.predict(X_test_scaled)
y_real_test = np.expm1(y_test)
y_real_pred = np.expm1(y_pred)
rmse_real = np.sqrt(mean_squared_error(y_real_test, y_real_pred))
print(f"Évaluation finale (réelle) — RMSE en euros : {rmse_real:.2f}")

# Sauvegarde du modèle
model.save("models/deep/deep_model.h5")
print("Modèle sauvegardé : models/deep/deep_model.h5")