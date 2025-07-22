# File: code/dl_predict_test.py

from dl_predict import predict_price_dl  # à adapter selon ton fichier réel

desc = "Auditer votre système qualité pour améliorer la fiabilité des processus et garantir la conformité aux normes."

fiabilites = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

for f in fiabilites:
    prix = predict_price_dl(desc, f)
    print(f"Fiabilité = {f*100:.0f}% → Prix prédit (DL) = {prix} €")
