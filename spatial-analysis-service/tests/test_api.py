from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "UP"}


def test_analyze_ok():
    pts = [{"lat": 45.2671 + i * 1e-5, "lon": 19.8335} for i in range(10)]
    body = {"sensors": [{"id": "S1", "lat": 45.30, "lon": 19.90}], "points": pts, "min_samples": 3}
    r = client.post("/spatial/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["total_points"] == 10
    assert len(data["clusters"]) == 1
    assert data["clusters"][0]["covered"] is False
    assert len(data["recommended_sensors"]) == 1


def test_invalid_latitude_is_rejected():
    r = client.post("/spatial/analyze", json={"points": [{"lat": 123, "lon": 19.8}]})
    assert r.status_code == 422


def test_invalid_eps_is_rejected():
    r = client.post("/spatial/analyze", json={"points": [], "eps_m": 0})
    assert r.status_code == 422


def test_too_many_points_is_rejected():
    pts = [{"lat": 45.0, "lon": 19.0}] * 10_001
    assert client.post("/spatial/analyze", json={"points": pts}).status_code == 422
