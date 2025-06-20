# app_gradio.py
"""
Interface Gradio pour la prédiction multi-sorties sur le jeu de données Fiverr.
Ce notebook/code illustre :
- **Bloc 3** : C2 (prétraitement), C3 (entraînement ML supervisé)
- **Bloc 5** : C1 (préparation données non structurées), C4 (déploiement MLOps)
- Avec mapping précis de chaque section aux compétences.
"""
import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# === Bloc 3 - C2 & Bloc 5 - C1 : Prétraitement et préparation des données ===
# 1. Chargement et préparation des préprocesseurs
DATA_PATH = os.path.join('data', 'fiverr_gigs', 'fiverr-data-gigs-cleaned.csv')
df = pd.read_csv(DATA_PATH, encoding='latin-1', low_memory=False)

# 1a. TF-IDF sur le titre (Bloc 5 C1 : texte -> vecteurs numériques)
vectoriseur_titre = TfidfVectorizer(max_features=2000, stop_words='english')
vectoriseur_titre.fit(df['Title'].fillna(''))

# 1b. Encodage one-hot du niveau du vendeur (Bloc 3 C2 : encodage catégoriel)
ohe_niveau = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_niveau.fit(df['Seller Level'].fillna('Inconnu').values.reshape(-1, 1))

# === Bloc 5 - C4 : Déploiement MLOps (chargement des modèles) ===
# 2. Chargement des modèles entraînés (Bloc 3 C3 : modèles supervisés sauvegardés)
MODELS_DIR = 'models'
modele_reg = joblib.load(os.path.join(MODELS_DIR, 'mor_rf.pkl'))  # RF pour prix & note
modele_clf = joblib.load(os.path.join(MODELS_DIR, 'level_clf.pkl'))  # LR pour niveau

# Historisation pour démonstration MLOps (suivi des prédictions)
history = []

# === Bloc 3 - C3 & Bloc 5 - C4 : Fonction de prédiction et pipeline complet ===
def predict(titre_service, niveau_vendeur, nombre_avis):
    """
    Prédit : prix, note moyenne et niveau du vendeur.
    - Bloc 3 C2 : prétraitement (TF-IDF, one-hot, numerique)
    - Bloc 3 C3 : inférence ML supervisé
    - Bloc 5 C4 : wrapping dans une API Gradio
    """
    # Prétraitement (Bloc 3 C2)
    vec_titre = vectoriseur_titre.transform([titre_service])
    vec_niveau = ohe_niveau.transform([[niveau_vendeur]])
    vec_avis = sparse.csr_matrix(np.array([[int(nombre_avis)]]))
    X = sparse.hstack([vec_titre, sparse.csr_matrix(vec_niveau), vec_avis], format='csr')

    # Inférence (Bloc 3 C3)
    prix_pred, note_pred = modele_reg.predict(X)[0]
    idx_niveau = modele_clf.predict(X)[0]
    niveau_pred = ohe_niveau.categories_[0][idx_niveau]

    # Arrondir résultats
    prix = round(float(prix_pred), 2)
    note = round(float(note_pred), 2)

    # Historisation (MLOps)
    history.append({'Titre': titre_service, 'Prix': prix, 'Note': note, 'Niveau': niveau_pred})
    df_hist = pd.DataFrame(history)

    # Visualisation (Bloc 5 C4)
    fig_prix, ax1 = plt.subplots()
    ax1.plot(df_hist.index + 1, df_hist['Prix'], marker='o')
    ax1.set_title('Historique des prix prédits')
    ax1.set_xlabel('Appel')
    ax1.set_ylabel('Prix (EUR)')
    plt.tight_layout()

    fig_note, ax2 = plt.subplots()
    ax2.plot(df_hist.index + 1, df_hist['Note'], marker='o', color='green')
    ax2.set_title('Historique des notes prédites')
    ax2.set_xlabel('Appel')
    ax2.set_ylabel('Note moyenne')
    plt.tight_layout()

    return prix, note, niveau_pred, df_hist, fig_prix, fig_note

# === Bloc 5 - C4 : Déploiement de l'interface Gradio ===
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Titre du service", placeholder="Ex: Je vais développer votre site web..."),
        gr.Dropdown(label="Niveau du vendeur", choices=ohe_niveau.categories_[0].tolist()),
        gr.Number(label="Nombre d'avis", value=0)
    ],
    outputs=[
        gr.Number(label="Prix prédit (EUR)"),
        gr.Number(label="Note moyenne prédite"),
        gr.Textbox(label="Niveau du vendeur prédit"),
        gr.Dataframe(label="Historique des prédictions", headers=['Titre', 'Prix', 'Note', 'Niveau']),
        gr.Plot(label="Graphique des prix prédits"),
        gr.Plot(label="Graphique des notes prédites")
    ],
    title="Prédiction multi-sorties Fiverr",
    description="Entrez le titre du service, le niveau du vendeur et le nombre d'avis pour prédire le prix, la note et le niveau, avec historique et graphiques."
)

if __name__ == '__main__':
    iface.launch()
