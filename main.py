# File: code/main.py

import os
import subprocess
import sys
import time
import webbrowser
import threading

# Chemins attendus des modèles
MODELS_REQUIRED = {
    "models/regression/gradient_boosting.pkl": "Modèle de régression (Gradient Boosting)",
    "models/regression/scaler.pkl": "Scaler pour la régression",
    "models/classification/decision_tree.pkl": "Modèle de classification (Decision Tree)",
    "models/classification/scaler.pkl": "Scaler pour la classification",
    "models/deep/deep_model.h5": "Modèle deep learning",
    "models/deep/scaler.pkl": "Scaler pour le deep learning",
}

def check_models():
    """Vérifie la présence de tous les fichiers modèles essentiels"""
    missing = []
    for path, label in MODELS_REQUIRED.items():
        if not os.path.exists(path):
            missing.append((path, label))
    return missing

def prompt_retrain():
    """Propose de relancer l’entraînement si des fichiers sont absents"""
    missing = check_models()
    if missing:
        print("Modèles manquants détectés :")
        for path, label in missing:
            print(f" - {label} ({path})")

        reponse = input("\nVoulez-vous entraîner les modèles maintenant ? (o/n) : ").strip().lower()
        if reponse == "o":
            subprocess.run([sys.executable, "code/train_all.py"])
        else:
            print("\nLancement interrompu car les modèles requis sont absents.")
            sys.exit(10)

def launch_apis():
    """Lance les deux APIs FastAPI (ML + DL) en parallèle"""
    print("Lancement des APIs FastAPI en arrière-plan...")

    # API ML — port 8001
    subprocess.Popen([sys.executable, "-m", "uvicorn", "code.ml_fastapi:app", "--port", "8001", "--reload"])

    # API DL — port 8000
    subprocess.Popen([sys.executable, "-m", "uvicorn", "code.dl_fastapi:app", "--port", "8000", "--reload"])

    # Attente courte pour laisser les serveurs démarrer
    time.sleep(10)

def launch_gradio():
    """Lance l’application Gradio et ouvre le navigateur"""
    print("Lancement de l’interface Gradio...")

    def open_browser():
        time.sleep(15)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser).start()
    subprocess.run([sys.executable, "code/app_gradio.py"])

if __name__ == "__main__":
    prompt_retrain()
    launch_apis()
    launch_gradio()
