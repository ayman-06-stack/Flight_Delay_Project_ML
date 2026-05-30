import pandas as pd
import numpy as np

def charger_et_nettoyer(chemin_csv):

    print("Chargement du dataset...")
    df = pd.read_csv(chemin_csv, low_memory=False)
    print(f"Shape initial : {df.shape}")

    # ── 1. Supprimer les vols annulés et détournés ──────────────────────────
    # Ces vols n'ont pas de retard mesurable → on les exclut
    df = df[df['CANCELLED'] == 0]
    df = df[df['DIVERTED'] == 0]
    print(f"Après suppression annulés/détournés : {df.shape}")

    # ── 2. Supprimer les colonnes inutiles ──────────────────────────────────
    colonnes_a_supprimer = [
        'CANCELLATION_REASON',  # vide car on a supprimé les annulés
        'TAIL_NUMBER',          # identifiant avion, pas prédictif
        'FLIGHT_NUMBER',        # numéro de vol, trop spécifique
        'WHEELS_OFF', 'WHEELS_ON',  # intermédiaires non utiles
        'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',   # causes de retard
        'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', # connues APRÈS le vol
        'WEATHER_DELAY'         # connue APRÈS le vol (data leakage !)
    ]
    df = df.drop(columns=colonnes_a_supprimer)

    # ── 3. Valeurs manquantes ───────────────────────────────────────────────
    print("\nValeurs manquantes avant traitement :")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # DEPARTURE_DELAY et ARRIVAL_DELAY : imputer par la médiane
    mediane_dep = df['DEPARTURE_DELAY'].median()
    mediane_arr = df['ARRIVAL_DELAY'].median()
    df['DEPARTURE_DELAY'].fillna(mediane_dep, inplace=True)
    df['ARRIVAL_DELAY'].fillna(mediane_arr, inplace=True)

    # DEPARTURE_TIME, ARRIVAL_TIME : supprimer les lignes manquantes
    df.dropna(subset=['DEPARTURE_TIME', 'ARRIVAL_TIME'], inplace=True)

    # Autres colonnes numériques : médiane
    colonnes_num = df.select_dtypes(include=[np.number]).columns
    df[colonnes_num] = df[colonnes_num].fillna(df[colonnes_num].median())

    print(f"\nValeurs manquantes après traitement : {df.isnull().sum().sum()}")

    # ── 4. Correction des types ─────────────────────────────────────────────
    df['MONTH']        = df['MONTH'].astype(int)
    df['DAY']          = df['DAY'].astype(int)
    df['DAY_OF_WEEK']  = df['DAY_OF_WEEK'].astype(int)
    df['YEAR']         = df['YEAR'].astype(int)

    # ── 5. Détection et traitement des outliers (Z-score) ───────────────────
    # Formule : Z = (x - μ) / σ  →  si |Z| > 3 : outlier
    from scipy import stats
    z_scores = np.abs(stats.zscore(df['DEPARTURE_DELAY']))
    avant = len(df)
    df = df[z_scores < 3]
    print(f"\nOutliers supprimés : {avant - len(df)} lignes")

    # ── 6. Créer la variable cible ──────────────────────────────────────────
    # y = 1 si retard >= 15 minutes, 0 sinon
    df['RETARD'] = (df['DEPARTURE_DELAY'] >= 15).astype(int)
    print(f"\nDistribution de la cible :")
    print(df['RETARD'].value_counts(normalize=True).round(3))

    # ── 7. Sauvegarder ──────────────────────────────────────────────────────
    df.to_csv('data_nettoyee.csv', index=False)
    print(f"\nDataset nettoyé sauvegardé : {df.shape}")
    return df


if __name__ == "__main__":
    df = charger_et_nettoyer('data/flights.csv')