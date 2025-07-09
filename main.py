# File: code/main.py

"""
## Lancement de l'application

Le fichier `main.py` constitue le **point d’entrée principal** pour tester l’ensemble du projet.

Il permet :

1. de vérifier la présence des modèles (régression, classification, deep learning),
2. d’entraîner automatiquement les modèles si besoin (`train_all.py`),
3. de lancer l’API REST avec FastAPI en arrière-plan,
4. de démarrer l’application Gradio pour tester les prédictions.

### Exécution

Dans un terminal (à la racine du projet), lancez simplement :

    python main.py
    
"""

import os
import subprocess
import sys
import time
import webbrowser

# Chemins attendus des modèles
MODELS_REQUIRED = {
    "models/regression/gradient_boosting.pkl": "Modèle de régression (gradient boosting)",
    "models/regression/scaler.pkl": "Scaler pour la régression",
    "models/classification/decision_tree.pkl": "Modèle de classification (decision tree)",
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
    """Propose de relancer la phase d'entraînement si des fichiers sont absents"""
    missing = check_models()
    if missing:
        print("Modèles manquants détectés :")
        for path, label in missing:
            print(f" - {label} ({path})")

        reponse = input("\nVoulez-vous entraîner les modèles maintenant ? (o/n) : ").strip().lower()
        if reponse == "o":
            subprocess.run([sys.executable, "code/train_all.py"])
        else:
            print("\nLe lancement est interrompu car les modèles requis sont absents.")
            sys.exit(1)

def launch_api():
    """Lance l’API FastAPI dans un sous-processus (non bloquant)"""
    print("Lancement de l’API FastAPI en arrière-plan...")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "code.api_fastapi:app", "--reload"])

    # Pause pour s’assurer que l’API a bien démarré avant Gradio
    time.sleep(2)

def launch_gradio():
    """Lance l’application Gradio et ouvre le navigateur"""
    print("Lancement de l’interface Gradio...")

    # Ouvrir l'URL dans le navigateur par défaut après un court délai
    def open_browser():
        time.sleep(2)  # Laisser Gradio démarrer
        webbrowser.open("http://127.0.0.1:7860")

    # Lancement du navigateur dans un thread séparé
    import threading
    threading.Thread(target=open_browser).start()

    # Démarrage de l'application Gradio (bloquant)
    subprocess.run([sys.executable, "code/app_gradio.py"])

if __name__ == "__main__":
    prompt_retrain()
    launch_api()
    launch_gradio()