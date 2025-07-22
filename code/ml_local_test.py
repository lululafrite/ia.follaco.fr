from ml_predict import predict_price, predict_tranche

desc = "Faire le ménage dans la maison, nettoyer les vitres, passer l'aspirateur et laver le sol. Faire la vaisselle et ranger la cuisine. Nettoyer les toilettes et la salle de bain. Ranger le salon et les chambres."

for f in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
    prix = predict_price(desc, f)
    print(f"Fiabilité = {f*100:.0f}% → Prix prédit = {prix} €")
    tranche = predict_tranche(desc, f)
    print(f"Fiabilité = {f*100:.0f}% → Tranche prédit = {tranche}")