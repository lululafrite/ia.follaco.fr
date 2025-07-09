# File: code/train_all.py

import subprocess
import sys
import os

# Liste des scripts à exécuter dans l’ordre logique
scripts = [
    "preprocessing.py",
    "training_regression.py",
    "training_classification.py",
    "training_deep.py"
]

# Répertoire courant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"\nExécution de : {script_name}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name} :")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    print("Lancement du pipeline complet d'entraînement...\n")
    for script in scripts:
        run_script(script)
    print("\nTous les modèles ont été entraînés avec succès.")