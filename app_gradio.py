import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# 1️⃣ Chargement du jeu de données nettoyé
DATA_PATH = os.path.join('data', 'fiverr_gigs', 'fiverr_cleaned.csv')
df = pd.read_csv(DATA_PATH, encoding='utf-8', low_memory=False)

# 2️⃣ Reconstruction des preprocessors (fit sur l'ensemble des données)
# — TF-IDF sur le titre du service
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf.fit(df['Description'].fillna(''))

# — One-Hot Encoding sur le niveau du vendeur
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe.fit(df['Niveau'].fillna('Inconnu').values.reshape(-1, 1))

# 3️⃣ Chargement des modèles entraînés
model_reg = joblib.load(os.path.join('models', 'mor_rf.pkl'))    # RandomForest multi-output
model_clf = joblib.load(os.path.join('models', 'level_clf.pkl')) # LogisticRegression

# 4️⃣ Fonction de prédiction
def predict(title, level, n_reviews):
    # a) Prétraitement
    v_title   = tfidf.transform([title])
    v_level   = ohe.transform([[level]])
    v_reviews = sparse.csr_matrix(np.array([[int(n_reviews)]]))
    X_input   = sparse.hstack([v_title, sparse.csr_matrix(v_level), v_reviews], format='csr')
    # b) Inférence
    prix_pred, note_pred = model_reg.predict(X_input)[0]
    idx_level = model_clf.predict(X_input)[0]
    level_pred = ohe.categories_[0][idx_level]
    # c) Formatage
    return round(float(prix_pred), 2), round(float(note_pred), 2), level_pred

# 5️⃣ Définition de l’interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Titre du service", placeholder="Ex: I will build your website..."),
        gr.Dropdown(label="Niveau du vendeur", choices=ohe.categories_[0].tolist()),
        gr.Number(label="Nombre d'avis", value=0)
    ],
    outputs=[
        gr.Number(label="Prix prédit (USD)"),
        gr.Number(label="Note moyenne prédite"),
        gr.Textbox(label="Niveau du vendeur prédit")
    ],
    title="Démo : Prédiction Fiverr Gigs",
    description="Entrez le titre du service, le niveau du vendeur et le nombre d'avis pour obtenir une prédiction du prix horaire (USD), de la note et du niveau via nos modèles ML."
)

if __name__ == "__main__":
    iface.launch()
