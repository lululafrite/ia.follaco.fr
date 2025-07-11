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

# Chargement des données préparées
df = pd.read_csv("data/fiverr_cleaned_transformed.csv")

# Embedding des descriptions
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
descriptions = df["Description"].astype(str).tolist()
embeddings = embedding_model.encode(descriptions)
embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

# Encodage one-hot du niveau du vendeur
niveau_encoded = pd.get_dummies(df["Niveau"], prefix="Niveau")

# Ajout de la variable Fiabilite
autres_features = df[["Fiabilite"]].reset_index(drop=True)

# Fusion des features
X = pd.concat([embed_df, niveau_encoded, autres_features], axis=1)
y = df["Prix"]

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Sauvegarde du modèle
model.save("models/deep/deep_model.h5")
print("Modèle sauvegardé : models/deep/deep_model.h5")