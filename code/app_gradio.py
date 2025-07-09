# File: code/app_gradio.py

import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib
import requests
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from predict import predict_price, predict_tranche

# Paramètres généraux
LOG_PATH = "flagged/log.csv"
API_URL = "http://127.0.0.1:8000/predict"

choices = {
    "Acceptable": 80,
    "Moyenne": 85,
    "Bonne": 90,
    "Très Bonne": 96,
    "Excellente": 99
}

niveau_mapping = ["Beginner", "Intermediate", "Expert"]
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
deep_model = None
scaler_dl = None

def load_deep_models():
    global deep_model, scaler_dl
    if deep_model is None:
        deep_model = load_model("models/deep/deep_model.h5")
        scaler_dl = joblib.load("models/deep/scaler.pkl")

def faire_une_prediction(description, niveau, use_predefined, fiabilite_percent, fiabilite_choix, modele):
    fiabilite = (choices[fiabilite_choix] if use_predefined else fiabilite_percent) / 100

    try:
        if modele == "ML - Local":
            prix = predict_price(description, fiabilite)
            tranche = predict_tranche(description, fiabilite)

        elif modele == "DL - Local":
            load_deep_models()
            emb = embedding_model.encode([description]).flatten()
            niveau_ohe = [1 if niveau == n else 0 for n in niveau_mapping]
            features = np.hstack([emb, niveau_ohe, [fiabilite]])
            features_scaled = scaler_dl.transform([features])
            prix = deep_model.predict(features_scaled)[0][0]
            tranche = "Non évaluée"

        elif modele == "API - FastAPI":
            response = requests.post(
                API_URL,
                json={"Description": description, "Niveau": niveau, "Fiabilite": fiabilite},
                timeout=5
            )
            response.raise_for_status()
            prix = response.json().get("prix_predit", -1)
            tranche = "Non évaluée"
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

with gr.Blocks() as iface:
    gr.Markdown(
        "Application de prédiction de prix d'un service Fiverr.\n"
        "Trois types de modèles disponibles : ML local, Deep Learning local, ou API REST.\n"
        "Lancez manuellement l’API si vous utilisez le mode API (commande : uvicorn api_fastapi:app --reload)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            description = gr.Textbox(label="Titre du service", value="Je fais le ménage")
            niveau = gr.Dropdown(label="Niveau du vendeur", choices=niveau_mapping, value="Beginner", visible=False)
            use_predefined = gr.Checkbox(label="Utiliser un niveau de fiabilité prédéfini", value=True)
            fiabilite_percent = gr.Slider(label="Fiabilité (%)", minimum=0, maximum=100, value=80, step=5, visible=False)
            fiabilite_choix = gr.Radio(label="Niveau de fiabilité", choices=list(choices.keys()), value="Acceptable", visible=True)
            modele = gr.Radio(label="Modèle à utiliser", choices=["ML - Local", "DL - Local", "API - FastAPI"], value="ML - Local")
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

        with gr.Column(scale=1):
            sortie_prix = gr.Textbox(label="Prix estimé")
            sortie_tranche = gr.Textbox(label="Tranche estimée")
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