"""HTTP-API för Saldes churn-modell."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn.data import FEATURES, skapa_features

MODEL_PATH = Path("outputs/modell.joblib")

app = FastAPI(title="Saldes Churn API", version="1.0.0")


class Kund(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    alder: int = Field(ge=18, le=100)
    manader_som_kund: int = Field(ge=0)
    manadskostnad: float = Field(ge=0)
    data_gb_per_manad: float = Field(ge=0)
    supportarenden_12m: int = Field(ge=0)
    region: str = Field(min_length=1)
    abonnemang: str = Field(min_length=1)


@lru_cache
def get_model():
    """Ladda modellen en gång och återanvänd den."""
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    """Kontrollera att modellen går att ladda."""
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Modellen är inte tillgänglig",
        ) from exc

    return {"status": "ok"}


@app.post("/predict")
def predict(kund: Kund):
    """Returnera sannolikheten för att kunden avslutar abonnemanget."""
    try:
        modell = get_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Modellen är inte tillgänglig",
        ) from exc

    df = pd.DataFrame([kund.model_dump()])
    X = skapa_features(df)[FEATURES]

    sannolikhet = float(modell.predict_proba(X)[0, 1])

    return {"churn_risk": round(sannolikhet, 4)}
