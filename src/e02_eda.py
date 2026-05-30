import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def faire_eda(df):

    print("=" * 50)
    print("ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 50)

    # ── 1. Statistiques descriptives ────────────────────────────────────────
    print("\nStatistiques de DEPARTURE_DELAY :")
    print(df['DEPARTURE_DELAY'].describe())

    # ── 2. Distribution des retards ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogramme des retards
    axes[0,0].hist(df['DEPARTURE_DELAY'].clip(-30, 120),
                   bins=60, color='steelblue', edgecolor='white')
    axes[0,0].axvline(15, color='red', linestyle='--', label='Seuil 15 min')
    axes[0,0].set_title('Distribution des retards au départ')
    axes[0,0].set_xlabel('Retard (minutes)')
    axes[0,0].legend()

    # Taux de retard par compagnie
    taux_compagnie = df.groupby('AIRLINE')['RETARD'].mean().sort_values(ascending=False)
    axes[0,1].bar(taux_compagnie.index, taux_compagnie.values, color='coral')
    axes[0,1].set_title('Taux de retard par compagnie')
    axes[0,1].set_ylabel('Taux de retard')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Taux de retard par mois
    taux_mois = df.groupby('MONTH')['RETARD'].mean()
    axes[1,0].plot(taux_mois.index, taux_mois.values,
                   marker='o', color='purple', linewidth=2)
    axes[1,0].set_title('Taux de retard par mois')
    axes[1,0].set_xlabel('Mois')
    axes[1,0].set_xticks(range(1,13))

    # Taux de retard par jour de la semaine
    jours = ['Lun','Mar','Mer','Jeu','Ven','Sam','Dim']
    taux_jour = df.groupby('DAY_OF_WEEK')['RETARD'].mean()
    axes[1,1].bar(jours, taux_jour.values, color='teal')
    axes[1,1].set_title('Taux de retard par jour')

    plt.tight_layout()
    plt.savefig('eda_visualisations.png', dpi=150)
    plt.show()
    print("Graphiques sauvegardés : eda_visualisations.png")

    # ── 3. Matrice de corrélation ────────────────────────────────────────────
    cols_corr = ['DEPARTURE_DELAY','DISTANCE','ELAPSED_TIME',
                 'AIR_TIME','TAXI_OUT','TAXI_IN','MONTH','DAY_OF_WEEK']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols_corr].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('data_nettoyee.csv')
    faire_eda(df)