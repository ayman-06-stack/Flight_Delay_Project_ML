from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.schemas import FlightInput, PredictionOutput
from api.predictor import FlightDelayPredictor

# ── Initialisation ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Flight Delay Predictor API",
    description="API de prédiction des retards de vols — Projet ML 2ème année GI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
predictor = FlightDelayPredictor()

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def interface(request: Request):
    """Interface web de démonstration"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Vérification que le serveur fonctionne"""
    return {"status": "ok", "modele": "XGBoost", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(vol: FlightInput):
    """
    Prédit si un vol sera en retard ou à l'heure.
    
    - **prediction** : 0 = à l'heure, 1 = en retard
    - **probabilite_retard** : probabilité entre 0 et 1
    - **niveau_risque** : Faible / Modéré / Élevé
    """
    try:
        result = predictor.predict(vol.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(vols: list[FlightInput]):
    """Prédit plusieurs vols en une seule requête"""
    if len(vols) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 vols par batch")
    return [predictor.predict(v.dict()) for v in vols]


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000)