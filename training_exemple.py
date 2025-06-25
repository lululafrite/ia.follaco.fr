# train_multioutput.py
"""
Entraînement de modèles multi-output sur le dataset Fiverr.
- Charge X (sparse) et y (price, rating, level)
- Supprime les exemples où price ou rating est NaN
- Split train/test
- Entraînement et comparaison de deux approches :
   1. Pipeline scikit-learn :
      - MultiOutputRegressor(RandomForestRegressor)
      - LogisticRegression pour la classification du level
   2. Modèle Keras multitâche (MLP partagé + têtes distinctes)
- Évaluation (MAE, RMSE, R² pour régressions; accuracy pour classification)
- Sauvegarde des meilleurs modèles
"""
import os
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import joblib


def load_data():
    dir_feat = os.path.join('data', 'fiverr_features')
    X = sparse.load_npz(os.path.join(dir_feat, 'X_features.npz'))
    y_price = np.load(os.path.join(dir_feat, 'y_price.npy'))
    y_rating = np.load(os.path.join(dir_feat, 'y_rating.npy'))
    y_level = np.load(os.path.join(dir_feat, 'y_level.npy'))
    # Combine price and rating into a 2D array for multi-output regressor
    y_reg = np.vstack([y_price, y_rating]).T
    # Remove rows with NaN in price or rating
    mask = ~np.isnan(y_reg).any(axis=1)
    X = X[mask]
    y_reg = y_reg[mask]
    y_level = y_level[mask]
    return X, y_reg, y_level

if __name__ == '__main__':
    X, y_reg, y_level = load_data()
    # Train/test split
    X_train, X_test, y_reg_train, y_reg_test, y_level_train, y_level_test = train_test_split(
        X, y_reg, y_level, test_size=0.2, random_state=42
    )

    # 1) Pipeline scikit-learn
    print("\n--- Entraînement scikit-learn MultiOutputRandomForest ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mor = MultiOutputRegressor(rf)
    mor.fit(X_train, y_reg_train)
    # Classification pour level
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_level_train)
    # Évaluation
    y_reg_pred = mor.predict(X_test)
    y_level_pred = clf.predict(X_test)
    print("Prix MAE:", mean_absolute_error(y_reg_test[:, 0], y_reg_pred[:, 0]))
    print("Prix RMSE:", np.sqrt(mean_squared_error(y_reg_test[:, 0], y_reg_pred[:, 0])))
    print("Prix R2:", r2_score(y_reg_test[:, 0], y_reg_pred[:, 0]))
    print("Note MAE:", mean_absolute_error(y_reg_test[:, 1], y_reg_pred[:, 1]))
    print("Note R2:", r2_score(y_reg_test[:, 1], y_reg_pred[:, 1]))
    print("Level accuracy:", accuracy_score(y_level_test, y_level_pred))
    # Sauvegarde scikit-learn
    os.makedirs('models', exist_ok=True)
    joblib.dump(mor, 'models/mor_rf.pkl')
    joblib.dump(clf, 'models/level_clf.pkl')
    print("Modèles scikit-learn sauvegardés dans /models.")

    # 2) Modèle Keras multitâche
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
    except ImportError:
        print("TensorFlow non installé, on passe le modèle Keras.")
    else:
        print("\n--- Entraînement Keras multi-tâche ---")
        # Split train into train+val for Keras
        X_tr, X_val, y_reg_tr, y_reg_val, y_level_tr, y_level_val = train_test_split(
            X_train, y_reg_train, y_level_train, test_size=0.1, random_state=42
        )
        # Convert sparse to dense for Keras
        X_tr = X_tr.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        n_features = X_tr.shape[1]
        inp = keras.Input(shape=(n_features,), dtype=tf.float32)
        x = layers.Dense(256, activation='relu')(inp)
        x = layers.Dense(128, activation='relu')(x)
        price_out = layers.Dense(1, name='price')(x)
        rating_out = layers.Dense(1, name='rating')(x)
        level_out = layers.Dense(len(np.unique(y_level_tr)), activation='softmax', name='level')(x)
        model = keras.Model(inputs=inp, outputs=[price_out, rating_out, level_out])
        model.compile(
            optimizer='adam',
            loss={'price': 'mse', 'rating': 'mse', 'level': 'sparse_categorical_crossentropy'},
            metrics={'price': 'mae', 'rating': 'mae', 'level': 'accuracy'}
        )
        # Entraînement
        model.fit(
            X_tr,
            [y_reg_tr[:, 0], y_reg_tr[:, 1], y_level_tr],
            validation_data=(X_val, [y_reg_val[:, 0], y_reg_val[:, 1], y_level_val]),
            epochs=10,
            batch_size=8,
            callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        )
        model.save('models/keras_multi_task.h5')
        print("Modèle Keras multitâche sauvegardé dans /models.")
