import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Données historiques moyennes par compagnie et aéroport
# (extraites de ton dataset pendant l'entraînement)
AIRLINE_AVG_DELAYS = {
    'NK': 18.2, 'UA': 14.1, 'F9': 13.8, 'WN': 12.4,
    'B6': 12.1, 'MQ': 11.5, 'VX': 10.9, 'AA': 10.2,
    'EV': 9.8,  'OO': 9.4,  'US': 9.1,  'DL': 8.7,
    'AS': 7.9,  'HA': 4.2
}

AIRPORT_AVG_DELAYS = {
    'ORD': 14.2, 'EWR': 13.8, 'SFO': 12.9, 'JFK': 12.1,
    'LAX': 10.5, 'ATL': 9.8,  'DFW': 9.2,  'MIA': 11.3,
    'BOS': 10.1, 'LGA': 13.5, 'DEN': 8.9,  'SEA': 8.2,
    'PHX': 7.8,  'MCO': 9.1,  'LAS': 8.5,  'HNL': 5.2
}

AIRLINE_LABEL_MAP  = {a: i for i, a in enumerate(sorted(AIRLINE_AVG_DELAYS.keys()))}
AIRPORT_CODES = sorted(set(list(AIRPORT_AVG_DELAYS.keys())))
AIRPORT_LABEL_MAP  = {a: i for i, a in enumerate(AIRPORT_CODES)}

class FlightDelayPredictor:
    
    def __init__(self):
        base = Path(__file__).parent.parent
        self.model  = joblib.load(base / 'models' / 'model_xgb.pkl')
        self.scaler = joblib.load(base / 'models' / 'scaler.pkl')
        print("Modèle XGBoost chargé avec succès")

    def _build_features(self, data: dict) -> np.ndarray:
        """Reconstruit exactement les 18 features du fichier 03"""
        
        heure = (data['SCHEDULED_DEPARTURE'] // 100)
        heure = max(0, min(23, heure))

        # Encodage cyclique
        heure_sin = np.sin(2 * np.pi * heure / 24)
        heure_cos = np.cos(2 * np.pi * heure / 24)
        mois_sin  = np.sin(2 * np.pi * data['MONTH'] / 12)
        mois_cos  = np.cos(2 * np.pi * data['MONTH'] / 12)

        # Variables binaires
        is_weekend     = int(data['DAY_OF_WEEK'] >= 6)
        is_peak_season = int(data['MONTH'] in [7, 8, 12])
        is_morning     = int(5 <= heure <= 11)
        is_evening     = int(17 <= heure <= 22)

        # Moyennes historiques
        airline_avg = AIRLINE_AVG_DELAYS.get(data['AIRLINE'], 10.0)
        airport_avg = AIRPORT_AVG_DELAYS.get(data['ORIGIN_AIRPORT'], 10.0)

        # Encodage label
        airline_enc = AIRLINE_LABEL_MAP.get(data['AIRLINE'], 0)
        origin_enc  = AIRPORT_LABEL_MAP.get(data['ORIGIN_AIRPORT'], 0)
        dest_enc    = AIRPORT_LABEL_MAP.get(data['DESTINATION_AIRPORT'], 0)

        features = np.array([[
            data['MONTH'],
            data['DAY_OF_WEEK'],
            data['DISTANCE'],
            data['SCHEDULED_TIME'],
            data.get('TAXI_OUT', 15.0),
            heure_sin, heure_cos,
            mois_sin,  mois_cos,
            is_weekend, is_peak_season, is_morning, is_evening,
            airline_avg, airport_avg,
            airline_enc, origin_enc, dest_enc
        ]])
        
        return features

    def predict(self, data: dict) -> dict:
        X = self._build_features(data)
        
        prob = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])

        prob_retard = round(float(prob[1]), 4)
        prob_heure  = round(float(prob[0]), 4)

        # Niveau de risque
        if prob_retard < 0.30:
            niveau = "Faible"
        elif prob_retard < 0.55:
            niveau = "Modéré"
        else:
            niveau = "Élevé"

        return {
            "prediction": pred,
            "label": "En retard" if pred == 1 else "À l'heure",
            "probabilite_retard": prob_retard,
            "probabilite_heure":  prob_heure,
            "niveau_risque": niveau,
            "details": {
                "compagnie": data['AIRLINE'],
                "trajet": f"{data['ORIGIN_AIRPORT']} → {data['DESTINATION_AIRPORT']}",
                "heure_depart": f"{data['SCHEDULED_DEPARTURE']:04d}",
                "distance_miles": data['DISTANCE']
            }
        }