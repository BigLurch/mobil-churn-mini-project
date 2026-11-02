"""Tester för Saldes HTTP-API."""

import numpy as np
from unittest.mock import Mock
from fastapi.testclient import TestClient
from churn.api import app, get_model

client = TestClient(app)

TESTKUND = {
    "alder": 35,
    "manader_som_kund": 24,
    "manadskostnad": 299.0,
    "data_gb_per_manad": 12.0,
    "supportarenden_12m": 2,
    "region": "Stockholm",
    "abonnemang": "Standard",
}


def test_health():
    """En tillgänglig modell ska ge HTTP 200."""
    app.dependency_overrides.clear()
    try:
        # Ersätt modellanropet med en kontrollerad testmodell.
        get_model.cache_clear()
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "churn.api.joblib.load"
        ) as load:
            load.return_value = Mock()
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        get_model.cache_clear()


def test_predict(monkeypatch):
    """En giltig kund ska ge en churn-risk."""
    import churn.api as api_module

    modell = Mock()
    modell.predict_proba.return_value = np.array([[0.25, 0.75]])
    monkeypatch.setattr(api_module, "get_model", lambda: modell)

    response = client.post("/predict", json=TESTKUND)

    assert response.status_code == 200
    assert response.json()["churn_risk"] == 0.75


def test_predict_invalid_data():
    """Negativ ålder ska avvisas."""
    felaktig_kund = {**TESTKUND, "alder": -5}

    response = client.post("/predict", json=felaktig_kund)

    assert response.status_code == 422
