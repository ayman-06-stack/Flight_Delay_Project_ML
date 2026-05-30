import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def entrainer_modeles(X, y):

    # ── Split train / test ───────────────────────────────────────────────────
    # 80% entraînement, 20% test — stratifié pour garder le ratio de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    # ── Normalisation (nécessaire pour la régression logistique) ────────────
    # Formule : x' = (x - μ) / σ
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)   # IMPORTANT : transform seulement, pas fit

    resultats = {}

    # ── MODÈLE 1 : Régression Logistique ────────────────────────────────────
    # ŷ = σ(θ₀ + θ₁x₁ + ... + θₙxₙ)   où  σ(z) = 1 / (1 + e^(-z))
    # Coût : J(θ) = -1/m Σ [y log(ŷ) + (1-y) log(1-ŷ)]
    print("\nEntraînement Régression Logistique...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)
    resultats['Logistic Regression'] = {
        'model': lr, 'X_test': X_test_sc, 'scaler': scaler
    }

    # ── MODÈLE 2 : Random Forest ─────────────────────────────────────────────
    # ŷ = mode(ŷ₁, ŷ₂, ..., ŷ_N)  — vote majoritaire de N arbres
    # Gini : 1 - Σ pₖ²   |   IG = H(parent) - Σ (|Dⱼ|/|D|) H(Dⱼ)
    print("Entraînement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,    # nombre d'arbres
        max_depth=10,        # profondeur max pour éviter overfitting
        min_samples_split=20,
        random_state=42,
        n_jobs=-1            # utiliser tous les CPU
    )
    rf.fit(X_train, y_train)
    resultats['Random Forest'] = {
        'model': rf, 'X_test': X_test, 'scaler': None
    }

    # ── MODÈLE 3 : XGBoost ───────────────────────────────────────────────────
    # ŷ⁽ᵗ⁾ = ŷ⁽ᵗ⁻¹⁾ + fₜ(x)   — ajout séquentiel d'arbres
    # Objectif : L(t) = Σ l(yᵢ, ŷᵢ) + Ω(fₜ)   où Ω = γT + ½λ||w||²
    print("Entraînement XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,   # η (eta) — pas d'apprentissage
        subsample=0.8,       # fraction de données par arbre
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    resultats['XGBoost'] = {
        'model': xgb, 'X_test': X_test, 'scaler': None
    }

    # ── Validation croisée (5-fold) ──────────────────────────────────────────
    # CV_score = (1/k) Σᵢ scoreᵢ
    print("\nValidation croisée (5-fold) :")
    for nom, info in resultats.items():
        X_cv = X_train_sc if info['scaler'] else X_train
        scores = cross_val_score(info['model'], X_cv, y_train,
                                  cv=5, scoring='f1', n_jobs=-1)
        print(f"  {nom} — F1 moyen : {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Sauvegarder les modèles ──────────────────────────────────────────────
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr,     'models/model_logistic.pkl')
    joblib.dump(rf,     'models/model_rf.pkl')
    joblib.dump(xgb,    'models/model_xgb.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModèles sauvegardés dans models/ (.pkl)")

    return resultats, X_test, X_test_sc, y_test


if __name__ == "__main__":
    df = pd.read_csv('data_features.csv')
    y = df['RETARD']
    X = df.drop(columns=['RETARD'])
    resultats, X_test, X_test_sc, y_test = entrainer_modeles(X, y)