# File: code/preprocessing.py

# === Import des bibliothèques nécessaires ===
import os
import re
import pandas as pd
import numpy as np
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from kaggle.api.kaggle_api_extended import KaggleApi

# === Téléchargement des stopwords français (pour nettoyage de texte) ===
nltk.download('stopwords')
english_stopwords = stopwords.words('english')

# === Authentification et téléchargement du fichier csv Kaggle ===
api = KaggleApi()
api.authenticate()
csvFile = "muhammadadiltalay/fiverr-data-gigs"
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)
api.dataset_download_files(csvFile, path=DATA_FOLDER, unzip=True)

# === Suppression du fichier nettoyé existant s’il existe déjà ===
file_path = os.path.join(DATA_FOLDER, "fiverr-data-gigs-cleaned.csv")
if os.path.exists(file_path):
    os.remove(file_path)

# === Chargement du fichier brut ===
df = pd.read_csv(os.path.join(DATA_FOLDER, "fiverr-data-gigs.csv"), encoding='utf-8', low_memory=False)

# === Renommage des colonnes principales pour clarté ===
df.rename(columns={'Title': 'Description', 'Title_URL': 'Lien', '_5fo9i5': 'Niveau', 'Price': 'Prix'}, inplace=True)

# === Suppression des lignes sans niveau vendeur valide ===
df = df[df['Niveau'] != "Clients"]

# === Encodage manuel des niveaux de vendeurs ===
df['Niveau'] = df['Niveau'].replace({
    "Level 1 Seller": 1,
    "Level 2 Seller": 2,
    "Top Rated Seller": 3
})

# === Nettoyage des prix : suppression de caractères, conversion en float ===
df['Prix'] = (
    df['Prix'].astype(str)
    .str.replace(r'[^\d,]', '', regex=True)
    .str.replace(',', '.', regex=False)
    .str.replace(r'\.(?=.*\.)', '', regex=True)
    .replace('', np.nan)
    .astype(float)
)

# === Extraction de la note moyenne depuis la colonne texte ===
df['Evaluation'] = df['gigrating'].str.extract(r'^([\d.]+)').astype(float)

# === Fonction d’extraction du nombre d’avis à partir d’un texte brut ===
def parse_review_count(s):
    if pd.isna(s): return np.nan
    match = re.search(r'\(([^)]+)\)', s)
    if not match: return np.nan
    val = match.group(1).lower().replace('+', '')
    return float(val.replace('k', '')) * 1000 if 'k' in val else float(val)

df['Rating_Count'] = df['gigrating'].apply(parse_review_count)

# === Suppression des colonnes inutiles ===
df.drop(columns=['gigrating', 'Lien'], inplace=True)

# === Fonction de nettoyage texte (accents, ponctuation, espaces) ===
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Description'] = df['Description'].apply(clean_text)

# === Encodage des niveaux manquants pour KNNImputer ===
le = LabelEncoder()
df['Niveau'] = df['Niveau'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan})
mask_notna = df['Niveau'].notna()
df.loc[mask_notna, 'Level_encoded'] = le.fit_transform(df.loc[mask_notna, 'Niveau'])

# === Imputation des colonnes numériques (dont niveau encodé) ===
cols_for_knn = ['Prix', 'Evaluation', 'Rating_Count', 'Level_encoded']
df_knn = df[cols_for_knn].copy()
imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=cols_for_knn, index=df.index)

# === Décodage du niveau à partir du label encodé imputé ===
encoded = df_knn_imputed['Level_encoded'].round().astype('Int64')
encoded.index = df.index
valid_range = list(range(len(le.classes_)))
mask_valid = encoded.isin(valid_range)

niveau_imputes = pd.Series(index=df.index, dtype='object')
encoded_valid = encoded[mask_valid].astype(int)
niveau_imputes.loc[mask_valid] = le.inverse_transform(encoded_valid.to_numpy())
niveau_imputes.loc[~mask_valid] = np.nan
df['Niveau'] = niveau_imputes
df.drop(['Level_encoded'], axis=1, inplace=True)

# === Imputation des autres colonnes numériques ===
cols_for_impute = ['Prix', 'Evaluation', 'Rating_Count']
df_impute = df[cols_for_impute].copy()
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_impute)
df[cols_for_impute] = df_imputed

# === Nettoyage final des chaînes de caractères ===
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# === Création d’un score de fiabilité (note moyenne pondérée par nb d’avis) ===
df['Fiabilite'] = (df['Evaluation'] / 5) * (1 - 1 / (1 + np.log1p(df['Rating_Count'])))
df.drop(['Evaluation', 'Rating_Count'], axis=1, inplace=True)

# === Nettoyage plus poussé de la description : suppression des débuts redondants ===
def clean_description(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', text)
    phrases_to_remove = [
        r"^i\s+will\s+do\s+", r"^i\s+will\s+create\s+", r"^i\s+will\s+be\s+your\s+",
        r"^i\s+will\s+write\s+", r"^i\s+will\s+provide\s+", r"^i\s+will\s+design\s+",
        r"^i\s+can\s+", r"^i\s+am\s+going\s+to\s+", r"^i\s+will\s+"
    ]
    for pattern in phrases_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    words = [word for word in text.split() if word not in english_stopwords]
    return " ".join(words).strip()

df["Description"] = df["Description"].astype(str).apply(clean_description)

# === Suppression des outliers sur la colonne Prix (IQR) ===
q1 = df["Prix"].quantile(0.25)
q3 = df["Prix"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["Prix"] >= lower_bound) & (df["Prix"] <= upper_bound)]
df["Fiabilite"] = df["Fiabilite"] * df["Niveau"].astype(float) # Ajustement de la fiabilité par niveau

# === Création du fichier pour le modèle ML ===
df_ml = df[["Description", "Prix", "Fiabilite"]].copy()
df_ml.to_csv("data/fiverr_cleaned_ml.csv", index=False, encoding="utf-8")
#df_ml.to_parquet("data/fiverr_cleaned_ml.parquet", index=False, engine="pyarrow", compression="snappy")

# === Création du fichier pour le modèle DL ===
#df_dl = df[["Description", "Niveau", "Fiabilite"]].copy()
df_dl = df[["Description", "Niveau", "Prix", "Fiabilite"]].copy()
df_dl["Prix_log"] = np.log1p(df["Prix"]) # Ajout du log du prix pour la modélisation

df_dl.to_csv("data/fiverr_cleaned_dl.csv", index=False, encoding="utf-8")
#df_dl.to_parquet("data/fiverr_cleaned_dl.parquet", index=False, engine="pyarrow", compression="snappy")