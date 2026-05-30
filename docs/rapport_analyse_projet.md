# Rapport d'Analyse Exhaustive : Prédiction des Retards des Vols

**Projet :** Utilisation de données historiques pour prédire si un vol sera en retard ou à l’heure.
**Domaine :** Machine Learning / Data Science

---

## 1. Analyse globale du projet

### 1.1. Objectif métier
L'objectif principal de ce projet est d'anticiper les retards de vols commerciaux. Ce système prédictif est un outil d'aide à la décision pour les compagnies aériennes, les gestionnaires d'aéroports et les passagers. Il permet d'estimer en amont la probabilité qu'un vol subisse un retard significatif (défini à 15 minutes ou plus).

### 1.2. Type de problème d'apprentissage
Il s'agit d'un problème d'**Apprentissage Supervisé** et plus spécifiquement de **Classification Binaire**. 
- **Entrée ($X$)** : Caractéristiques du vol prévu (date, compagnie, origine, destination, distance, etc.).
- **Sortie ($y$)** : Une variable discrète valant `1` (En retard $\ge$ 15 min) ou `0` (À l'heure).

*Note : La prédiction du nombre exact de minutes de retard aurait relevé de la **Régression**, mais la binarisation par seuil simplifie le problème et répond souvent mieux à la question métier "Dois-je m'inquiéter d'un retard ?".*

### 1.3. Enjeux métier
- **Pour les passagers :** Améliorer l'expérience client en fournissant des alertes anticipées, permettant d'ajuster leurs plannings (ex: correspondances).
- **Pour les compagnies :** Optimiser la rotation des équipages et des appareils, réduire les pénalités liées aux retards, ajuster dynamiquement les ressources au sol.
- **Pour les aéroports :** Mieux gérer l'allocation des portes d'embarquement et le trafic sur les pistes (taxi-out / taxi-in).

---

## 2. Analyse complète de la structure du code

Le projet est structuré de manière modulaire, respectant les bonnes pratiques d'ingénierie logicielle avec une séparation claire des étapes du cycle de vie du modèle.

### 2.1. `e01_nettoyage.py`
- **Rôle :** Préparation initiale des données.
- **Fonction `charger_et_nettoyer(chemin_csv)` :** 
  - *Entrée :* Chemin vers le dataset brut (`flights.csv`).
  - *Processus :* Suppression des vols annulés/détournés, élimination des variables causant un **data leakage** (ex: `WEATHER_DELAY`, connues *après* le vol), traitement des valeurs manquantes par imputation (médiane), et filtrage des valeurs aberrantes (outliers) via le Z-Score. Enfin, binarisation de la cible `RETARD`.
  - *Sortie :* Dataset nettoyé (`data_nettoyee.csv`).

### 2.2. `e02_eda.py` (Exploratory Data Analysis)
- **Rôle :** Analyse exploratoire et visualisation de la donnée.
- **Fonction `faire_eda(df)` :**
  - *Entrée :* Le dataset nettoyé.
  - *Processus :* Génère des statistiques descriptives et crée des graphiques (histogrammes de distribution, taux de retard par compagnie, par mois, par jour). Produit également une matrice de corrélation (`sns.heatmap`) pour identifier les relations linéaires entre variables numériques.
  - *Sorties :* Fichiers PNG des visualisations (`eda_visualisations.png`, `correlation_matrix.png`).

### 2.3. `e03_feature_engineering.py`
- **Rôle :** Création de variables prédictives enrichies pour aider l'algorithme.
- **Fonction `creer_features(df)` :**
  - *Processus :* Encodage cyclique (sin/cos) du mois et de l'heure. Variables catégorielles extraites (week-end, saison haute, matin, soir). Création de moyennes historiques (`Target Encoding`) par compagnie et aéroport. Encodage numérique (`LabelEncoder`) des variables textuelles (`AIRLINE`, etc.).
  - *Sorties :* Matrice de features $X$, vecteur cible $y$, et sauvegarde dans `data_features.csv`.

### 2.4. `e04_modelisation.py`
- **Rôle :** Entraînement des algorithmes de Machine Learning.
- **Fonction `entrainer_modeles(X, y)` :**
  - *Processus :* Séparation des données en Train/Test (80/20) avec stratification. Standardisation des données (`StandardScaler`). Entraînement de trois modèles concurrents : Régression Logistique, Random Forest, XGBoost. Validation croisée à 5 plis pour évaluer la robustesse.
  - *Sorties :* Objets de modèles ajustés et exportés en `.pkl` via `joblib`.

### 2.5. `e05_evaluation.py`
- **Rôle :** Évaluation des performances sur l'ensemble de test invisible.
- **Fonction `evaluer_modeles(...)` :**
  - *Processus :* Calcule Accuracy, Precision, Recall, F1, et AUC. Génère les matrices de confusion et trace les courbes ROC.
  - *Sorties :* Comparatif final exporté en `resultats_evaluation.csv` et graphes dans `evaluation_resultats.png`.

### 2.6. Fichiers d'Interface et de Production
- **`main.py` :** Script chef d'orchestre qui exécute séquentiellement les étapes 1 à 5.
- **`app.py` :** Application web backend utilisant **FastAPI**. Expose un point d'accès web HTML (racine `/`) et des API REST (`/predict`, `/predict/batch`) pour inférer de nouvelles données.
- **`predictor.py` :** Classe `FlightDelayPredictor` qui encapsule la logique d'inférence de production. Elle charge le modèle `XGBoost`, le scaler, et reconstruit **exactement** le même pipeline de feature engineering (dictionnaires moyens codés en dur) pour générer la prédiction.
- **`schemas.py` :** Définit les contrats de données (Data Validation) via **Pydantic** pour l'API FastAPI, garantissant que l'entrée de l'utilisateur est saine et possède les bons types.

**Flux d'exécution global :** Données brutes $\rightarrow$ Nettoyage $\rightarrow$ EDA $\rightarrow$ Ingénierie des caractéristiques $\rightarrow$ Entraînement $\rightarrow$ Évaluation $\rightarrow$ Export Modèle $\rightarrow$ Déploiement FastAPI.

---

## 3. Pipeline Machine Learning détaillé

### 3.1. Collecte / Chargement des données
- **Format / Taille :** Fichier CSV (potentiellement très volumineux, au vu des ~800 Mo de `data_features.csv`).
- **Signification métier :** Le jeu de données répertorie le planning des vols (`SCHEDULED_DEPARTURE`), les aéroports (`ORIGIN`, `DESTINATION`), les compagnies (`AIRLINE`), et la mesure réelle de l'opération (retard en minutes `DEPARTURE_DELAY`).

### 3.2. Nettoyage des données
1. **Élimination des cas non pertinents :** Vols annulés (`CANCELLED=1`) ou détournés (`DIVERTED=1`).
2. **Prévention du Data Leakage :** Suppression draconienne des causes de retard (`WEATHER_DELAY`, `LATE_AIRCRAFT_DELAY`) car ces informations ne sont connues *qu'à la fin du vol*. Les inclure donnerait 100% de précision artificielle.
3. **Valeurs manquantes :** Imputation par la médiane pour les retards restants, suppression des lignes sans heure de départ/arrivée, et médiane pour les autres numériques.
4. **Outliers :** Utilisation du **Z-Score** mathématique $Z = \frac{x - \mu}{\sigma}$. Tout vol s'éloignant de plus de 3 écarts-types de la moyenne ($\approx 99.7\%$ de la distribution normale) est exclu pour ne pas biaiser le modèle.

### 3.3. Feature Engineering
- **Encodage Cyclique :** L'heure de départ et le mois sont convertis en coordonnées circulaires (`sin` et `cos`). C'est crucial car l'heure 23 et l'heure 0 sont adjacentes temporellement, mais très éloignées numériquement. Les vecteurs sinus/cosinus règlent cela.
- **Variables temporelles métier :** Création des booléens explicites `IS_WEEKEND`, `IS_PEAK_SEASON` (été + décembre), `IS_MORNING`, `IS_EVENING`.
- **Target Encoding (Moyenne Historique) :** Création de `AIRLINE_AVG_DELAY` et `AIRPORT_AVG_DELAY`. On remplace le nom de la compagnie par la *moyenne statistique* de ses retards.
- **Scaling :** Standardisation $x' = \frac{x - \mu}{\sigma}$ appliquée uniquement aux modèles y étant sensibles (Régression logistique).

### 3.4. Split des données
- **Méthode :** `train_test_split` à $80\%$ d'entraînement et $20\%$ de test.
- **Stratification :** L'argument `stratify=y` est utilisé, ce qui est une excellente pratique. Il garantit que le ratio de vols retardés/à l'heure est identique dans les sets d'entraînement et de test, évitant les biais sur des classes déséquilibrées.

---

## 4. Analyse mathématique et théorique des modèles ML

### 4.1. Logistic Regression (Régression Logistique)
- **Principe :** Modèle linéaire utilisé pour la classification. Il trouve un hyperplan séparateur optimal entre les deux classes en utilisant la fonction sigmoïde pour écraser la sortie continue entre 0 et 1 (probabilité).
- **Formule de prédiction :** $\hat{y} = \sigma(\theta_0 + \theta_1 x_1 + \dots + \theta_n x_n)$
- **Fonction sigmoïde :** $\sigma(z) = \frac{1}{1 + e^{-z}}$ (où $z$ est la combinaison linéaire des variables).
- **Fonction de coût (Log-Loss) :** 
  $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$
  *Explication : Pénalise fortement le modèle s'il prédit une probabilité proche de 0 alors que la vérité est 1 (et inversement).*
- **Pourquoi ce choix :** Sert de `Baseline` (modèle de référence) rapide, très interprétable, permettant de vérifier si le problème peut être résolu de manière purement linéaire.

### 4.2. Random Forest (Forêt Aléatoire)
- **Principe :** Algorithme de *Bagging* (Bootstrap Aggregating). Construit de multiples arbres de décision ($N=100$) sur des sous-échantillons aléatoires des données et des features.
- **Mécanique :** Chaque arbre procède à des séparations binaires maximisant le gain d'information (baisse de l'impureté de Gini). La prédiction finale est le **vote majoritaire** de tous les arbres.
- **Impureté de Gini :** $Gini = 1 - \sum_{k=1}^{K} p_k^2$
  *Où $p_k$ est la proportion de la classe $k$ dans un nœud. Plus Gini est bas, plus le nœud est pur.*
- **Pourquoi ce choix :** Modèle très robuste au surapprentissage grâce à l'aléatoire, qui gère nativement les relations non linéaires et qui ne nécessite pas de scaling des données.

### 4.3. XGBoost (eXtreme Gradient Boosting)
- **Principe :** Algorithme de *Boosting*. Contrairement à Random Forest où les arbres sont indépendants, XGBoost construit les arbres **séquentiellement**. Chaque nouvel arbre tente de corriger les *erreurs (résidus)* de l'arbre précédent.
- **Formule de prédiction :** $\hat{y}_i^{(t)} = \sum_{k=1}^{t} f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i)$
- **Fonction Objectif à minimiser :** 
  $Obj(t) = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$
  *Où $L$ est la perte logistique, et $\Omega$ est le terme de régularisation (pénalité de complexité) :* $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ ($T$: nb feuilles, $w$: poids des feuilles).
- **Pourquoi ce choix :** Généralement l'algorithme "State of the Art" sur données tabulaires. Il optimise la descente de gradient de manière mathématiquement agressive et gère très bien les relations complexes.

---

## 5. Entraînement et optimisation

- **Hyperparamètres :** 
  - *Random Forest :* `max_depth=10` (contrôle la profondeur pour éviter l'overfitting), `min_samples_split=20` (un nœud doit avoir 20 observations pour être divisé).
  - *XGBoost :* `learning_rate=0.1` (pas de descente de gradient conservateur), `max_depth=6`, sous-échantillonnage de features/lignes `subsample=0.8, colsample_bytree=0.8` (injection de hasard type Random Forest dans le Boosting pour la robustesse).
- **Validation Croisée (Cross-Validation) :** Utilisation d'un 5-fold CV pour s'assurer que le modèle ne surapprend pas sur un subset particulier de données d'entraînement. La métrique suivie est le `F1-Score`.

---

## 6. Évaluation des performances

Les modèles sont évalués sur le set de test avec 5 métriques clés. 
Soit $TP$ (Vrai Positif), $TN$ (Vrai Négatif), $FP$ (Faux Positif, fausse alerte), $FN$ (Faux Négatif, retard raté) :

1. **Accuracy (Exactitude) :** $\frac{TP + TN}{TP + TN + FP + FN}$
   - *Interprétation :* Pourcentage global de prédictions justes. Souvent trompeur si le dataset est déséquilibré (ex: 80% de vols à l'heure, prédire "Toujours à l'heure" donne 80% d'accuracy).
2. **Precision :** $\frac{TP}{TP + FP}$
   - *Interprétation :* Quand le modèle crie "Ce vol va être en retard !", à quelle fréquence a-t-il raison ? Crucial si déclencher une alerte a un coût élevé.
3. **Recall (Rappel / Sensibilité) :** $\frac{TP}{TP + FN}$
   - *Interprétation :* Sur tous les vols qui ont *réellement* eu du retard, combien le modèle a-t-il réussi à intercepter ? Souvent la métrique reine ici : on veut rater le moins de retards possibles.
4. **F1-Score :** $2 \times \frac{Precision \times Recall}{Precision + Recall}$
   - *Interprétation :* La moyenne harmonique entre Précision et Rappel. Offre un compromis strict (un modèle avec une Précision de 100% et un Rappel de 1% aura un très mauvais F1).
5. **ROC-AUC (Area Under Curve) :**
   - *Interprétation :* Evalue la capacité du modèle à classer les probabilités. Un AUC de 0.5 équivaut à lancer une pièce. 0.84 (XGBoost ici) signifie qu'il a 84% de chances de donner un score plus élevé à un vol aléatoirement retardé qu'à un vol aléatoirement à l'heure.

---

## 7. Analyse des résultats (Critique)

D'après le code (ligne 25 de `main.py`), les AUC attendus sont : LogReg (0.72) < Random Forest (0.81) < XGBoost (0.84).

- **Interprétation :** Le problème est modérément non-linéaire, ce qui explique pourquoi l'arbre de décision (RF) détruit la Régression Logistique. XGBoost vient gratter les derniers pourcentages grâce au boosting itératif.
- **Risque d'Overfitting vs Underfitting :** Les paramètres stricts (`max_depth=6` pour XGB, `10` pour RF) montrent une bonne gestion de l'overfitting. Si le F1 du Random Forest chute radicalement sur le set de test par rapport au train, c'est que les arbres mémorisaient les feuilles.
- **Faiblesses du modèle :** Les retards de vols sont souvent dictés par la météo et les problèmes mécaniques imprévisibles le matin même. Sans météo en temps réel, le modèle atteint un plafond de verre (AUC de 0.84 est excellent avec ces données strictes).

---

## 8. Déploiement / Application

L'application passe la barrière du notebook expérimental pour aller vers la production.

1. **Framework backend :** **FastAPI**, tournant via le serveur Asynchrone **Uvicorn**. Très moderne, rapide et documenté automatiquement (Swagger).
2. **Composant Predictor :** La classe orientée objet `FlightDelayPredictor` lit en mémoire `model_xgb.pkl` une seule fois au démarrage (Singleton) pour des inférences très rapides.
3. **Workflow Utilisateur :**
   - Une requête HTTP `POST /predict` est envoyée avec un payload JSON respectant `FlightInput` (Mois, Compagnie, Aéroport...).
   - Pydantic valide les données (ex: Heure entre 0 et 2359).
   - L'API transmet à `predictor.predict()`.
   - Les features sont transformées (re-calcul des sinus, mapping dans les dicos de moyennes...).
   - XGBoost `.predict_proba()` renvoie le pourcentage.
   - Le serveur retourne une classification métier (Niveau de risque : Faible, Modéré, Élevé) en JSON.

---

## 9. Améliorations possibles & Critiques de l'existant

Bien que ce code soit de très bonne qualité académique, certains aspects demandent correction avant la mise en production professionnelle (MLOps) :

### 9.1. Critique Majeure de Code : Data Leakage dans le Target Encoding
**ATTENTION :** Dans `e03_feature_engineering.py`, les variables `AIRLINE_AVG_DELAY` sont calculées avec un `.groupby('AIRLINE')['DEPARTURE_DELAY'].mean()` sur **l'ensemble total du dataset (df)**, AVANT le fractionnement Train/Test du fichier `e04`.
- **Problème :** Le jeu de Test a "fuité" dans le jeu de Train. L'information des retards à prédire dans le futur a été utilisée pour calculer les moyennes.
- **Solution :** Calculer ces dictionnaires moyens **uniquement** sur les indices d'entraînement (`X_train`), puis `.map()` ces valeurs sur l'entraînement et le test.

### 9.2. Gestion du dictionnaire d'Inférence
Dans `predictor.py`, les dictionnaires `AIRLINE_AVG_DELAYS` sont **codés en dur** dans le script.
- **Problème :** C'est une dette technique. Si le modèle est ré-entraîné sur de nouvelles données, le développeur doit copier-coller manuellement les valeurs dans `predictor.py`.
- **Solution :** Exporter un fichier `category_means.json` à la fin de l'étape 3/4, et le charger via `json.load()` dans le constructeur de FastAPI. Cela permet de ré-entrainer le modèle dynamiquement sans modifier le code source de l'API.

### 9.3. Améliorations ML
- **Hyperparameter Tuning Automatique :** Le code utilise des paramètres choisis statiquement. L'intégration de `GridSearchCV` ou de bibliothèques bayésiennes comme `Optuna` permettrait de chercher mathématiquement les meilleurs paramètres (learning rate, depth).
- **Imbalance Management :** Ajouter l'argument `scale_pos_weight` dans XGBoost, ou utiliser la technique SMOTE pour sur-échantillonner artificiellement les cas de vols retardés (la classe minoritaire).

### 9.4. Ajout de Features Externes (Production)
- **API Météo :** Relier dynamiquement le pipeline à un historique Météo de la NOAA (température, neige, visibilité) au point de départ et d'arrivée. C'est le facteur manquant principal.
- **Chain Delay :** Un avion enchaîne plusieurs vols dans la journée. Si le vol de 10h est en retard de 3h, le vol de 14h sera retardé. Il faudrait suivre le `TAIL_NUMBER` (numéro de l'appareil) sur l'axe chronologique de la même journée. Le script l'a malheureusement supprimé lors du nettoyage initial.

---
*Fin de l'audit technique. Ce projet démontre une compréhension solide et avancée du cycle complet d'un projet de Data Science : de l'ingénierie de la donnée à l'API de déploiement.*
