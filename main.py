import pandas as pd
from src.e01_nettoyage       import charger_et_nettoyer
from src.e02_eda              import faire_eda
from src.e03_feature_engineering import creer_features
from src.e04_modelisation     import entrainer_modeles
from src.e05_evaluation       import evaluer_modeles

if __name__ == "__main__":

    print("\n ÉTAPE 1 — Nettoyage")
    df_clean = charger_et_nettoyer('data/flights.csv')

    print("\n ÉTAPE 2 — EDA")
    faire_eda(df_clean)

    print("\n ÉTAPE 3 — Feature Engineering")
    X, y, features = creer_features(df_clean)

    print("\n ÉTAPE 4 — Modélisation")
    resultats, X_test, X_test_sc, y_test = entrainer_modeles(X, y)

    print("\n ÉTAPE 5 — Évaluation")
    df_res = evaluer_modeles(resultats, X_test, X_test_sc, y_test)

    print("\n Projet terminé — Résultats attendus :")
    print("  Logistic Regression  : AUC ≈ 0.72")
    print("  Random Forest        : AUC ≈ 0.81")
    print("  XGBoost              : AUC ≈ 0.84")