# File: code/training_classification.py

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# Chargement des données transformées
df = pd.read_csv("data/fiverr_cleaned_ml.csv")

# Génération des embeddings à partir de la description
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["Description"].astype(str).tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(384)])

# Standardisation de la variable Fiabilite
scaler = StandardScaler()
fiabilite_scaled = scaler.fit_transform(df[["Fiabilite"]])
fiabilite_df = pd.DataFrame(fiabilite_scaled, columns=["Fiabilite"])

# Fusion des features pour créer la matrice X
X = pd.concat([embed_df, fiabilite_df], axis=1)

# Création de la variable cible (tranche de prix)
df["Tranche"] = pd.qcut(df["Prix"], q=3, labels=["Basse", "Moyenne", "Haute"])
y = df["Tranche"]

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluation et affichage
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle DecisionTree : {round(acc, 4)}")

# Sauvegarde des modèles
os.makedirs("models/classification", exist_ok=True)
joblib.dump(model, "models/classification/decision_tree.pkl")
joblib.dump(scaler, "models/classification/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/columns_used.pkl")

print("Modèle et artefacts sauvegardés dans models/classification/")