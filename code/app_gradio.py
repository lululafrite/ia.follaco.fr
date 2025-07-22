# code/app_gradio.py

import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib
import requests
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from ml_predict import predict_price, predict_tranche

# === Paramètres ===
LOG_PATH = "flagged/log.csv"
API_DL_URL = "http://127.0.0.1:8000/predict"
API_ML_URL_PRICE = "http://127.0.0.1:8001/predict_price"
API_ML_URL_TRANCHE = "http://127.0.0.1:8001/predict_tranche"

choices = {
    "Acceptable": 80,
    "Moyenne": 85,
    "Bonne": 90,
    "Très Bonne": 96,
    "Excellente": 99
}

niveau_mapping = {"Débutant": 1, "Confirmé": 2, "Expert": 3}
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
deep_model = None
scaler_dl = None

def load_deep_models():
    global deep_model, scaler_dl
    if deep_model is None:
        deep_model = load_model("models/deep/deep_model.h5")
        scaler_dl = joblib.load("models/deep/scaler.pkl")

def faire_une_prediction(description, niveau_label, use_predefined, fiabilite_percent, fiabilite_choix, modele):
    fiabilite = (choices[fiabilite_choix] if use_predefined else fiabilite_percent) / 100
    niveau = niveau_mapping.get(niveau_label, 1)

    try:
        if modele == "ML - Local":
            prix = predict_price(description, fiabilite)
            tranche = predict_tranche(description, fiabilite)

        elif modele == "DL - Local":
            load_deep_models()
            emb = embedding_model.encode([description]).flatten()
            niveau_ohe = [1 if niveau == n else 0 for n in [1, 2, 3]]
            features = np.hstack([emb, niveau_ohe, [fiabilite]])
            features_scaled = scaler_dl.transform([features])
            prix = round(deep_model.predict(features_scaled)[0][0] * 10, 2)
            tranche = "Non évaluée"

        elif modele == "DL - FastAPI":
            response = requests.post(
                API_DL_URL,
                json={"description": description, "niveau": niveau, "fiabilite": fiabilite},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            prix = round(data.get("prix", -1) * 10, 2)
            tranche = data.get("tranche", "Non évaluée")

        elif modele == "ML - FastAPI":
            prix = -1
            tranche = "Non évaluée"

            # Appel API pour le prix
            response_price = requests.post(
                API_ML_URL_PRICE,
                json={"description": description, "fiabilite": fiabilite},
                timeout=5
            )
            response_price.raise_for_status()
            prix = response_price.json().get("prix", -1)

            # Appel API pour la tranche
            response_tranche = requests.post(
                API_ML_URL_TRANCHE,
                json={"description": description, "fiabilite": fiabilite},
                timeout=5
            )
            response_tranche.raise_for_status()
            tranche = response_tranche.json().get("tranche", "Non évaluée")

        else:
            return "Modèle inconnu", ""

        return round(prix, 2), tranche

    except Exception as e:
        return f"Erreur : {str(e)}", ""

def enregistrer_log(description, use_predefined, fiabilite_percent, fiabilite_choix, prix, tranche, modele):
    fiabilite = choices[fiabilite_choix] if use_predefined else fiabilite_percent
    log_data = {
        "Description": description,
        "Fiabilité (%)": fiabilite,
        "Prix prédit (€)": prix,
        "Tranche prédite": tranche,
        "Modèle utilisé": modele
    }
    df_log = pd.DataFrame([log_data])
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if os.path.exists(LOG_PATH):
        df_log.to_csv(LOG_PATH, mode="a", index=False, header=False)
    else:
        df_log.to_csv(LOG_PATH, index=False)

    return "Signalement enregistré avec succès."

# Interface Gradio
with gr.Blocks() as iface:
    gr.Markdown(
        "Application de prédiction de prix d'un service Fiverr."
    )

    with gr.Row():
        with gr.Column(scale=1):
            description = gr.Textbox(label="Description du service",
                                      value="Développer un site web de e-commerce avec une interface utilisateur intuitive, un système de paiement sécurisé et une gestion efficace des stocks. Le site doit être optimisé pour le référencement et compatible avec les appareils mobiles",
                                      lines=2, placeholder="Auditer votre système qualité")
            use_predefined = gr.Checkbox(label="Utiliser un niveau de fiabilité prédéfini", value=True, visible=False)
            fiabilite_percent = gr.Slider(label="Fiabilité (%)", minimum=0, maximum=100, value=80, step=5, visible=False)
            fiabilite_choix = gr.Radio(label="Niveau de fiabilité", choices=list(choices.keys()), value="Acceptable", visible=True)
            modele = gr.Radio(
                label="Modèle à utiliser",
                choices=["ML - Local", "ML - FastAPI", "DL - Local", "DL - FastAPI"],
                value="ML - Local"
            )
            niveau = gr.Dropdown(label="Niveau du vendeur", choices=list(niveau_mapping.keys()), value="Débutant", visible=False)
            bouton_predire = gr.Button("Estimer le prix")

            def sync_slider_with_radio(choix):
                return gr.update(value=choices[choix])
            fiabilite_choix.change(sync_slider_with_radio, inputs=fiabilite_choix, outputs=fiabilite_percent)

            def toggle_inputs(use_predef):
                return {
                    fiabilite_percent: gr.update(visible=not use_predef),
                    fiabilite_choix: gr.update(visible=use_predef)
                }
            use_predefined.change(toggle_inputs, inputs=use_predefined, outputs=[fiabilite_percent, fiabilite_choix])

            def toggle_niveau(model_choice):
                return gr.update(visible=model_choice in ["DL - Local", "DL - FastAPI"])
            modele.change(toggle_niveau, inputs=modele, outputs=niveau)

        # === Colonne de droite ===
        with gr.Column(scale=1):
            # Champs résultat initiaux
            sortie_prix = gr.Textbox(label="Prix estimé en €/h",  value="Attente de prédiction", visible=False)
            def toggle_prix_visibility(model_choice):
                return gr.update(visible=model_choice in ["DL - Local", "DL - FastAPI"])
            modele.change(toggle_prix_visibility, inputs=modele, outputs=sortie_prix)

            sortie_tranche = gr.Textbox(label="Tendance des prix sur le marché",  value="Attente de prédiction", visible=True)
            def toggle_tranche_visibility(model_choice):
                return gr.update(visible=model_choice in ["ML - Local", "ML - FastAPI"])
            modele.change(toggle_tranche_visibility, inputs=modele, outputs=sortie_tranche)

            # Masquage du champ niveau si modèle non-DL
            def toggle_niveau(model_choice):
                return gr.update(visible=model_choice in ["DL - Local", "DL - FastAPI"])
            modele.change(toggle_niveau, inputs=modele, outputs=niveau)

            # Masquage dynamique du champ tranche selon modèle
            def toggle_tranche_visibility(model_choice):
                return gr.update(
                    visible=model_choice in ["ML - Local", "ML - FastAPI"],
                    value="Attente de prédiction"
                )
            modele.change(toggle_tranche_visibility, inputs=modele, outputs=sortie_tranche)

            # Masquage dynamique du champ prix selon modèle (il est toujours affiché ici)
            def reset_sortie_prix(model_choice):
                return gr.update(value="Attente de prédiction")
            modele.change(reset_sortie_prix, inputs=modele, outputs=sortie_prix)

            bouton_signaler = gr.Button("Ajouter au fichier log.csv")
            confirmation = gr.Textbox(label="Confirmation", visible=False)

            bouton_predire.click(
                fn=faire_une_prediction,
                inputs=[description, niveau, use_predefined, fiabilite_percent, fiabilite_choix, modele],
                outputs=[sortie_prix, sortie_tranche]
            )

            bouton_signaler.click(
                fn=enregistrer_log,
                inputs=[description, use_predefined, fiabilite_percent, fiabilite_choix, sortie_prix, sortie_tranche, modele],
                outputs=confirmation
            )


iface.launch()