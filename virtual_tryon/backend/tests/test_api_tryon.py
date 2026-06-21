from __future__ import annotations

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
    assert "garment" in response.json()["detail"].lower()


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
