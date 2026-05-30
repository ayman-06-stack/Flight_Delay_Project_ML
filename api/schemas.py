from pydantic import BaseModel, Field
from typing import Optional

class FlightInput(BaseModel):
    """Données d'entrée pour la prédiction d'un vol"""
    
    MONTH: int          = Field(..., ge=1, le=12,   description="Mois (1-12)")
    DAY_OF_WEEK: int    = Field(..., ge=1, le=7,    description="Jour semaine (1=Lun, 7=Dim)")
    AIRLINE: str        = Field(...,                description="Code compagnie ex: AA, DL, UA")
    ORIGIN_AIRPORT: str = Field(...,                description="Code aéroport départ ex: LAX")
    DESTINATION_AIRPORT: str = Field(...,           description="Code aéroport arrivée ex: JFK")
    SCHEDULED_DEPARTURE: int = Field(..., ge=0, le=2359, description="Heure départ ex: 1430")
    DISTANCE: float     = Field(..., gt=0,          description="Distance en miles")
    SCHEDULED_TIME: float = Field(..., gt=0,        description="Durée prévue en minutes")
    TAXI_OUT: Optional[float] = Field(default=15.0, description="Temps taxi (défaut 15 min)")

    class Config:
        json_schema_extra = {
            "example": {
                "MONTH": 6,
                "DAY_OF_WEEK": 5,
                "AIRLINE": "NK",
                "ORIGIN_AIRPORT": "LAX",
                "DESTINATION_AIRPORT": "JFK",
                "SCHEDULED_DEPARTURE": 1430,
                "DISTANCE": 2475.0,
                "SCHEDULED_TIME": 320.0,
                "TAXI_OUT": 15.0
            }
        }

class PredictionOutput(BaseModel):
    """Résultat de la prédiction"""
    prediction: int            # 0 = à l'heure, 1 = en retard
    label: str                 # "À l'heure" ou "En retard"
    probabilite_retard: float  # entre 0 et 1
    probabilite_heure: float
    niveau_risque: str         # "Faible", "Modéré", "Élevé"
    details: dict              # features utilisées