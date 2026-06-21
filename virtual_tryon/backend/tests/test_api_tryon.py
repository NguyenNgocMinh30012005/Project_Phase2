from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_health_returns_ok(client):
    api = TestClient(client)
    response = api.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "idm_vton" in payload["models"]


def test_tryon_reject_missing_person(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={"garment_top": png_file("top.png")},
    )
    assert response.status_code in {400, 422}


def test_tryon_reject_no_garment(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={"person_image": png_file("person.png")},
    )
    assert response.status_code == 400
    assert "garment" in response.json()["error"]["message"].lower()


def test_tryon_accepts_upper_body_request(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "true", "repair_mode": "true"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result_url"]
    assert payload["debug"]["mask_url"]


def test_tryon_api_error_shape(monkeypatch, png_file):
    monkeypatch.setenv("TRYON_ENGINE", "idm_vton")
    from app.core.config import clear_settings_cache
    from app.engines.idm_vton_engine import IDMVTonEngine
    from app.services.container import clear_container_cache
    from app.utils.errors import ModelUnavailableError
    import app.main as main_module

    def fail_run(self, inputs):
        raise ModelUnavailableError("IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl")

    monkeypatch.setattr(IDMVTonEngine, "run", fail_run)
    clear_settings_cache()
    clear_container_cache()
    reloaded = importlib.reload(main_module)
    api = TestClient(reloaded.app)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert "IDM-VTON" in payload["error"]
    assert "Traceback" not in payload["error"]

    monkeypatch.setenv("TRYON_ENGINE", "mock")
    clear_settings_cache()
    clear_container_cache()
    importlib.reload(main_module)
