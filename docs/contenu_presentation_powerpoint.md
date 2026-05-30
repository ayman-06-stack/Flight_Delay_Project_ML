# Prédiction des Retards de Vols Commerciaux
**Classification Binaire — Apprentissage Supervisé — Déploiement FastAPI**

---

## 1. Introduction & Contexte Projet
- **Présentation du projet et objectifs métier** : Anticiper les retards pour optimiser les opérations aéroportuaires et fluidifier l'expérience passager.
- **Problématique : Enjeux socio-économiques des retards aériens** : Coûts financiers majeurs pour les compagnies, impact écologique et insatisfaction client.
- **Cadrage technique : Apprentissage supervisé et classification binaire** : Prédire si un vol sera à l'heure (Classe 0) ou en retard (Classe 1) à partir de données historiques.

---

## 2. Analyse & Préparation des Données (Pipeline ETL)
- **Analyse Exploratoire (EDA)** : Étude des corrélations et statistiques descriptives pour identifier les facteurs influents (compagnie, aéroport de départ).
- **Nettoyage et Outliers** : Traitement des valeurs manquantes et élimination des valeurs aberrantes à l'aide du Z-Score.
  - *Formule Z-Score* : `Z = (x - μ) / σ`
- **Focus Sécurité : Identification et élimination du Data Leakage** : Suppression stricte des variables post-vol (comme le temps de retard réel en minutes ou les causes météo confirmées).

---

## 3. Feature Engineering & Représentation Mathématique
- **Encodage cyclique des variables temporelles (Sin/Cos)** : Conservation de la continuité logique du temps (les mois et les heures tournent en boucle).
  - *Formules* : `sin(2π × valeur / max)` et `cos(2π × valeur / max)`
- **Enrichissement** : Création de variables métier (saisonnalité, indicateur week-end) et application du Target Encoding pour les aéroports.
- **Transformation : Standardisation et mise à l'échelle** : Mise sur la même échelle de toutes les features numériques.
  - *Formule Standardisation* : `x' = (x - μ) / σ`

---

## 4. Architecture des Modèles & Fondements Théoriques
- **Baseline : Régression Logistique**
  - *Formule sigmoïde* : `σ(z) = 1 / (1 + e^(-z))`
  - *Fonction de coût* : Log-Loss
- **Approche Ensemble (Bagging) : Random Forest**
  - Mesure du gain d'information.
  - *Impureté de Gini* : `Gini = 1 - Σ(pk²)`
- **Approche Séquentielle (Boosting) : XGBoost**
  - Optimisation de la perte de façon séquentielle.
  - *Formule XGBoost* : `ŷ(t) = ŷ(t-1) + f_t(x)`
  - *Terme de régularisation* : `Ω(f) = γT + ½λΣwj²`

---

## 5. Stratégie d'Entraînement et Validation
- **Séparation stratifiée (80/20)** : Gestion du déséquilibre de classes pour garantir la même proportion de retards dans les sets de Train et Test.
- **Optimisation des hyperparamètres** : Recherche des meilleurs paramètres (Learning rate, profondeur maximale des arbres).
- **Validation croisée K-Fold** : Évaluation sur K sous-ensembles pour assurer la robustesse du modèle et éviter l'overfitting.

---

## 6. Évaluation Multicritères des Performances
- **Analyse des métriques** : En présence de classes déséquilibrées, l'Accuracy est trompeuse. Le F1-Score et l'AUC sont plus fiables.
  - *Formules* : Accuracy, Precision, Recall, F1, AUC
- **Interprétation des résultats** : Comparatif net entre la baseline LogReg (0.72) et le modèle avancé XGBoost (0.84).
- **Analyse d'erreurs** : Utilisation des matrices de confusion et des courbes ROC pour évaluer les faux positifs et faux négatifs.

---

## 7. Industrialisation & Déploiement (Mise en Production)
- **Architecture de l'API : FastAPI** : Serveur performant avec validation stricte des contrats de données grâce à Pydantic.
- **Design Pattern : Singleton** : Implémentation de la classe `FlightDelayPredictor` pour charger le modèle en mémoire une seule fois.
- **Interface & Inférence** : 
  - Endpoint de prédiction unitaire (un seul vol).
  - Endpoint par lot (batch) pour traiter des listes entières de plannings (CSV/JSON).

---

## 8. Conclusion & Perspectives
- **Synthèse et limites actuelles** : Succès de XGBoost pour la modélisation, mais le modèle souffre d'une dépendance aux facteurs météo externes impossibles à prévoir longtemps à l'avance.
- **Pistes d'amélioration** : 
  - Intégration d'APIs météo en temps réel.
  - Mise en place d'un pipeline MLOps pour surveiller la dérive du modèle (Data Drift) et automatiser le ré-entraînement.
