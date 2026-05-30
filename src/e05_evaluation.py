import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    accuracy_score, precision_score, recall_score
)

def evaluer_modeles(resultats, X_test, X_test_sc, y_test):

    print("=" * 55)
    print("ÉVALUATION DES MODÈLES")
    print("=" * 55)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    couleurs = {'Logistic Regression': 'blue',
                'Random Forest': 'green',
                'XGBoost': 'orange'}

    metriques_finales = {}

    for i, (nom, info) in enumerate(resultats.items()):
        model  = info['model']
        X_ev   = X_test_sc if info['scaler'] else X_test
        y_pred = model.predict(X_ev)
        y_prob = model.predict_proba(X_ev)[:, 1]

        # ── Métriques ──────────────────────────────────────────────────────
        # Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        # Précision = TP / (TP + FP)
        # Rappel    = TP / (TP + FN)
        # F1        = 2 × (Précision × Rappel) / (Précision + Rappel)
        # AUC-ROC   = aire sous la courbe ROC

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        metriques_finales[nom] = {
            'Accuracy': acc, 'Précision': prec,
            'Rappel': rec, 'F1': f1, 'AUC': auc
        }

        print(f"\n{'─'*40}")
        print(f"{nom}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Précision : {prec:.4f}")
        print(f"  Rappel    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=['À l\'heure', 'En retard']))

        # ── Matrice de confusion ───────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[0, i],
                    xticklabels=['À l\'heure', 'En retard'],
                    yticklabels=['À l\'heure', 'En retard'])
        axes[0, i].set_title(f'Confusion — {nom}')
        axes[0, i].set_ylabel('Réel')
        axes[0, i].set_xlabel('Prédit')

        # ── Courbe ROC ─────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[1, i].plot(fpr, tpr, color=couleurs[nom],
                        lw=2, label=f'AUC = {auc:.3f}')
        axes[1, i].plot([0,1],[0,1], 'k--', lw=1)
        axes[1, i].set_title(f'ROC — {nom}')
        axes[1, i].set_xlabel('FPR')
        axes[1, i].set_ylabel('TPR')
        axes[1, i].legend()

    plt.tight_layout()
    import os
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/evaluation_resultats.png', dpi=150)
    plt.show()

    # ── Tableau comparatif final ───────────────────────────────────────────
    df_resultats = pd.DataFrame(metriques_finales).T.round(4)
    print("\n" + "="*55)
    print("TABLEAU COMPARATIF FINAL")
    print(df_resultats.to_string())

    meilleur = df_resultats['F1'].idxmax()
    print(f"\nMeilleur modèle (F1) : {meilleur}")

    df_resultats.to_csv('resultats_evaluation.csv')
    return df_resultats


if __name__ == "__main__":
    import joblib, pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data_features.csv')
    y  = df['RETARD']
    X  = df.drop(columns=['RETARD'])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler   = joblib.load('models/scaler.pkl')
    X_test_sc = scaler.transform(X_test)

    resultats = {
        'Logistic Regression': {'model': joblib.load('models/model_logistic.pkl'),
                                 'X_test': X_test_sc, 'scaler': scaler},
        'Random Forest':       {'model': joblib.load('models/model_rf.pkl'),
                                 'X_test': X_test, 'scaler': None},
        'XGBoost':             {'model': joblib.load('models/model_xgb.pkl'),
                                 'X_test': X_test, 'scaler': None},
    }
    evaluer_modeles(resultats, X_test, X_test_sc, y_test)