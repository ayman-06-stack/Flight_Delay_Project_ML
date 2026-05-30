import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def creer_features(df):

    print("Feature Engineering en cours...")

    # ── 1. Encodage cyclique de l'heure ─────────────────────────────────────
    # Formule : sin(2π × heure / 24) et cos(2π × heure / 24)
    # Pourquoi ? 23h et 0h sont proches mais 23 et 0 sont loin numériquement
    df['HEURE_DEP'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int)
    df['HEURE_DEP'] = df['HEURE_DEP'].clip(0, 23)

    df['HEURE_SIN'] = np.sin(2 * np.pi * df['HEURE_DEP'] / 24)
    df['HEURE_COS'] = np.cos(2 * np.pi * df['HEURE_DEP'] / 24)

    # ── 2. Encodage cyclique du mois ─────────────────────────────────────────
    df['MOIS_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MOIS_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)

    # ── 3. Variables calendaires ─────────────────────────────────────────────
    df['IS_WEEKEND']      = (df['DAY_OF_WEEK'] >= 6).astype(int)
    df['IS_PEAK_SEASON']  = (df['MONTH'].isin([7, 8, 12])).astype(int)
    df['IS_MORNING']      = (df['HEURE_DEP'].between(5, 11)).astype(int)
    df['IS_EVENING']      = (df['HEURE_DEP'].between(17, 22)).astype(int)

    # ── 4. Retard moyen historique par compagnie ─────────────────────────────
    # Formule : d̄_airline = (1/n) Σ delay_i
    retard_moyen_airline = df.groupby('AIRLINE')['DEPARTURE_DELAY'].mean()
    df['AIRLINE_AVG_DELAY'] = df['AIRLINE'].map(retard_moyen_airline)

    # ── 5. Retard moyen historique par aéroport d'origine ───────────────────
    retard_moyen_airport = df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].mean()
    df['AIRPORT_AVG_DELAY'] = df['ORIGIN_AIRPORT'].map(retard_moyen_airport)

    # ── 6. Encodage Label des variables catégorielles ───────────────────────
    le = LabelEncoder()
    df['AIRLINE_ENC']      = le.fit_transform(df['AIRLINE'].astype(str))
    df['ORIGIN_ENC']       = le.fit_transform(df['ORIGIN_AIRPORT'].astype(str))
    df['DEST_ENC']         = le.fit_transform(df['DESTINATION_AIRPORT'].astype(str))

    # ── 7. Sélection des features finales ────────────────────────────────────
    features = [
        'MONTH', 'DAY_OF_WEEK', 'DISTANCE',
        'SCHEDULED_TIME', 'TAXI_OUT',
        'HEURE_SIN', 'HEURE_COS',
        'MOIS_SIN', 'MOIS_COS',
        'IS_WEEKEND', 'IS_PEAK_SEASON', 'IS_MORNING', 'IS_EVENING',
        'AIRLINE_AVG_DELAY', 'AIRPORT_AVG_DELAY',
        'AIRLINE_ENC', 'ORIGIN_ENC', 'DEST_ENC'
    ]

    X = df[features]
    y = df['RETARD']

    print(f"Features créées : {len(features)}")
    print(f"Shape X : {X.shape}")
    print(f"Distribution y : {y.value_counts().to_dict()}")

    # Sauvegarder
    df[features + ['RETARD']].to_csv('data_features.csv', index=False)
    return X, y, features


if __name__ == "__main__":
    df = pd.read_csv('data_nettoyee.csv')
    X, y, features = creer_features(df)