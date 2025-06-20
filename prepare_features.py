# prepare_features.py
"""
Préparation des features pour le dataset Fiverr multi-output.
- Charge le CSV nettoyé (fiverr-data-gigs-cleaned.csv)
- Nettoie et encode les colonnes numériques et catégorielles
- Transforme le titre en représentation TF-IDF
- Concatène toutes les features en une matrice X
- Extrait les cibles y_price, y_rating, y_level
- Sauvegarde X et y dans des fichiers .npz (sparse) et .npy
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 1. Chargement du DataFrame
FIVERR_PATH = os.path.join('data', 'fiverr_gigs', 'fiverr-data-gigs-cleaned.csv')
df = pd.read_csv(FIVERR_PATH, encoding='latin-1', low_memory=False)

# 2. Nettoyage et parsing
# 2.1. Nombre d'avis
# Convertit "1k+" -> 1000, "2" -> 2, etc.
def parse_reviews(x):
    if isinstance(x, str) and 'k' in x:
        return int(x.lower().replace('k+', '000').replace('k', '000'))
    try:
        return int(x)
    except:
        return 0

df['num_reviews'] = df['Number of Reviewers'].apply(parse_reviews)

# 2.2. Seller Level one-hot
levels = df['Seller Level'].fillna('Unknown').values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=True)
X_level = ohe.fit_transform(levels)

# 3. TF-IDF sur le titre
titles = df['Title'].fillna('').tolist()
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_title = tfidf.fit_transform(titles)

# 4. Features numériques
X_num = sparse.csr_matrix(df[['num_reviews']].values)

# 5. Concaténation de toutes les features
# [TF-IDF titre | one-hot level | num_reviews]
X = sparse.hstack([X_title, X_level, X_num], format='csr')

# 6. Extraction des cibles
# y_price: float
y_price = df['Price (USD)'].values.astype(np.float32)
# y_rating: float
y_rating = df['Average Rating'].values.astype(np.float32)
# y_level: encode level as integer label
level_labels = ohe.categories_[0]
# Map each Seller Level to its index in categories_
label_map = {lvl: idx for idx, lvl in enumerate(level_labels)}
y_level = df['Seller Level'].fillna('Unknown').map(label_map).values.astype(np.int32)

# 7. Sauvegarde des matrices et vecteurs
OUT_DIR = os.path.join('data', 'fiverr_features')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Features sparse
sparse.save_npz(os.path.join(OUT_DIR, 'X_features.npz'), X)
# Cibles
def save_array(arr, name):
    np.save(os.path.join(OUT_DIR, name), arr)

save_array(y_price, 'y_price.npy')
save_array(y_rating, 'y_rating.npy')
save_array(y_level, 'y_level.npy')

print(f"✅ Features saved in {OUT_DIR}:")
print(" - X_features.npz (shape =", X.shape, ")")
print(" - y_price.npy", y_price.shape)
print(" - y_rating.npy", y_rating.shape)
print(" - y_level.npy", y_level.shape)
