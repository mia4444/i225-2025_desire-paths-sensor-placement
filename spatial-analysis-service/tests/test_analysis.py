import numpy as np
import pytest

from app import analysis

CENTER = (45.2671, 19.8335)  # Novi Sad (approximate)

def offset(lat, lon, dn_m, de_m):
    """Shift a point by dn_m metres north and de_m metres east."""
    dlat = np.degrees(dn_m / analysis.EARTH_RADIUS_M)
    dlon = np.degrees(de_m / (analysis.EARTH_RADIUS_M * np.cos(np.radians(lat))))
    return lat + dlat, lon + dlon




def blob(center, n, spread_m, seed):
    rng = np.random.default_rng(seed)
    return [offset(center[0], center[1], rng.normal(0, spread_m), rng.normal(0, spread_m)) for _ in range(n)]

def test_haversine_one_degree_of_latitude_is_about_111_km():
    d, _ = analysis.nearest_distances_m([(1.0, 0.0)], [(0.0, 0.0)])
    assert d[0] == pytest.approx(111_195, rel=0.001)

def test_no_sensors_gives_infinite_distance():
    d, idx = analysis.nearest_distances_m([CENTER], [])
    assert np.isinf(d[0]) and idx[0] == -1

def test_dbscan_finds_two_clusters_and_noise():
    a = blob(CENTER, 40, 30, 1)
    b = blob(offset(*CENTER, 2000, 0), 40, 30, 2)
    far_noise = [offset(*CENTER, 5000, 5000)]
    labels = analysis.cluster_points(a + b + far_noise, eps_m=150, min_samples=5)
    assert len(set(labels) - {-1}) == 2
    assert labels[-1] == -1

def test_uncovered_cluster_gets_recommendation_and_coverage_rises():
    covered_hotspot = blob(CENTER, 40, 30, 1)
    desert_hotspot = blob(offset(*CENTER, 2000, 0), 60, 30, 2)
    sensors = [("S1", *offset(*CENTER, 20, 0))]  # right next to the first hotspot
    res = analysis.analyze(sensors, covered_hotspot + desert_hotspot, 150, 5, 200)

    assert len(res["clusters"]) == 2
    assert [c["covered"] for c in res["clusters"]].count(True) == 1
    assert len(res["recommended_sensors"]) == 1
    assert res["recommended_sensors"][0]["supporting_points"] == 60
    assert res["coverage"]["before_pct"] < 50
    assert res["coverage"]["after_pct"] > 90


def test_empty_points():
    res = analysis.analyze([("S1", *CENTER)], [], 150, 5, 200)
    assert res["total_points"] == 0 and res["clusters"] == []
    assert res["coverage"] == {"before_pct": 0.0, "after_pct": 0.0}


def test_no_sensors_every_cluster_is_uncovered():
    res = analysis.analyze([], blob(CENTER, 30, 30, 3), 150, 5, 200)
    assert res["clusters"][0]["nearest_sensor_id"] is None
    assert len(res["recommended_sensors"]) == 1
