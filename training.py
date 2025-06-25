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
    """
    Charge les features et les cibles depuis le dossier data/fiverr_features :
    - X_features.npz  : matrice creuse des features
    - y_prix.npy      : prix
    - y_evaluation.npy: note moyenne
    - y_niveau.npy    : niveau (étiquettes catégorielles)
    Retourne X (csr_matrix), y_reg (array 2D [prix, évaluation]), y_niveau (array 1D).
    Supprime les lignes où prix ou évaluation est NaN.
    """
    dir_feat = os.path.join('data', 'fiverr_features')
    X = sparse.load_npz(os.path.join(dir_feat, 'X_features.npz'))
    y_prix       = np.load(os.path.join(dir_feat, 'y_prix.npy'))
    y_evaluation = np.load(os.path.join(dir_feat, 'y_evaluation.npy'))
    y_niveau     = np.load(os.path.join(dir_feat, 'y_niveau.npy'))
    # Concatène prix et évaluation pour le MultiOutputRegressor
    y_reg = np.vstack([y_prix, y_evaluation]).T
    # Filtre les lignes sans NaN
    mask = ~np.isnan(y_reg).any(axis=1)
    return X[mask], y_reg[mask], y_niveau[mask]

if __name__ == "__main__":
    # ─── 1. Chargement des données ───────────────────────────────────────────────
    X, y_reg, y_niveau = load_data()
    print(f"Total samples after filtering : {X.shape[0]}")

    # ─── 2. Split train/test ───────────────────────────────────────────────────
    X_train, X_test, y_reg_train, y_reg_test, y_niveau_train, y_niveau_test = train_test_split(
        X, y_reg, y_niveau,
        test_size=0.2,
        random_state=42
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # ─── 3. Entraînement scikit-learn ──────────────────────────────────────────
    # 3.1 Régression multi-sorties (prix & évaluation)
    rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mor = MultiOutputRegressor(rf)
    mor.fit(X_train, y_reg_train)

    # 3.2 Classification pour le niveau
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_niveau_train)

    # 3.3 Évaluation
    y_reg_pred    = mor.predict(X_test)
    y_niveau_pred = clf.predict(X_test)

    print("=== Résultats scikit-learn ===")
    print("Prix MAE :", mean_absolute_error(y_reg_test[:, 0], y_reg_pred[:, 0]))
    print("Prix RMSE:", np.sqrt(mean_squared_error(y_reg_test[:, 0], y_reg_pred[:, 0])))
    print("Prix R²  :", r2_score(y_reg_test[:, 0], y_reg_pred[:, 0]))
    print("Note MAE :", mean_absolute_error(y_reg_test[:, 1], y_reg_pred[:, 1]))
    print("Note R²  :", r2_score(y_reg_test[:, 1], y_reg_pred[:, 1]))
    print("Level accuracy :", accuracy_score(y_niveau_test, y_niveau_pred))

    # 3.4 Sauvegarde des modèles scikit-learn
    os.makedirs('models', exist_ok=True)
    joblib.dump(mor, 'models/mor_rf.pkl')
    joblib.dump(clf, 'models/level_clf.pkl')
    print("✅ Modèles scikit-learn sauvegardés dans /models.")

    # ─── 4. (Optionnel) Entraînement Keras multi-tâche ─────────────────────────
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
    except ImportError:
        print("⚠️ TensorFlow non installé, on skippe le modèle Keras.")
    else:
        print("\n--- Entraînement Keras multi-tâche ---")
        # Split train → train+val
        X_tr, X_val, y_reg_tr, y_reg_val, y_niv_tr, y_niv_val = train_test_split(
            X_train, y_reg_train, y_niveau_train,
            test_size=0.1, random_state=42
        )
        # Passage en dense pour Keras
        X_tr  = X_tr .toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        n_features = X_tr.shape[1]

        inp = keras.Input(shape=(n_features,), dtype=tf.float32)
        x   = layers.Dense(256, activation='relu')(inp)
        x   = layers.Dense(128, activation='relu')(x)
        price_out  = layers.Dense(1, name='price')(x)
        rating_out = layers.Dense(1, name='rating')(x)
        level_out  = layers.Dense(len(np.unique(y_niv_tr)),
                                  activation='softmax', name='level')(x)
        model = keras.Model(inputs=inp, outputs=[price_out, rating_out, level_out])
        model.compile(
            optimizer='adam',
            loss={'price':'mse', 'rating':'mse', 'level':'sparse_categorical_crossentropy'},
            metrics={'price':'mae', 'rating':'mae', 'level':'accuracy'}
        )
        model.fit(
            X_tr,
            [y_reg_tr[:,0], y_reg_tr[:,1], y_niv_tr],
            validation_data=(X_val, [y_reg_val[:,0], y_reg_val[:,1], y_niv_val]),
            epochs=10,
            batch_size=8,
            callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        )
        model.save('models/keras_multi_task.h5')
        print("✅ Modèle Keras multitâche sauvegardé dans /models.")
