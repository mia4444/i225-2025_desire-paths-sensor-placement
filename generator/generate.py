import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

EARTH_RADIUS_M = 6_371_000.0
CENTER = (45.2671, 19.8335)  # approximate centre of Novi Sad

# Hotspots as (north_m, east_m, number_of_points, spread_m) relative to CENTER.
# First 'COVERED_HOTSPOTS' are placed on top of a sensor, the rest in the gaps of the grid.
HOTSPOTS = [
    (0, 0, 120, 40),
    (800, 800, 150, 45),
    (-400, 400, 140, 40),
    (400, -400, 110, 35),
    (-800, -400, 130, 45),
    (1200, 0, 90, 40),
]
GRID_SPACING_M = 800
GRID_HALF = 2  # grid of (2*2+1)^2 = 25 sensors


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised with NumPy)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def offset(lat, lon, north_m, east_m):
    dlat = np.degrees(north_m / EARTH_RADIUS_M)
    dlon = np.degrees(east_m / (EARTH_RADIUS_M * np.cos(np.radians(lat))))
    return lat + dlat, lon + dlon


def make_sensors() -> pd.DataFrame:
    rows = []
    k = 1
    for i in range(-GRID_HALF, GRID_HALF + 1):
        for j in range(-GRID_HALF, GRID_HALF + 1):
            lat, lon = offset(*CENTER, i * GRID_SPACING_M, j * GRID_SPACING_M)
            rows.append({"id": f"S{k:02d}", "lat": lat, "lon": lon})
            k += 1
    return pd.DataFrame(rows)


def make_points(rng: np.random.Generator, n_noise: int) -> pd.DataFrame:
    chunks = []
    for north, east, n, spread in HOTSPOTS:
        dn = rng.normal(north, spread, n)
        de = rng.normal(east, spread, n)
        lat, lon = offset(*CENTER, dn, de)
        chunks.append(pd.DataFrame({"lat": lat, "lon": lon, "kind": "hotspot"}))
    extent = GRID_HALF * GRID_SPACING_M + 400
    lat, lon = offset(*CENTER, rng.uniform(-extent, extent, n_noise), rng.uniform(-extent, extent, n_noise))
    chunks.append(pd.DataFrame({"lat": lat, "lon": lon, "kind": "noise"}))
    return pd.concat(chunks, ignore_index=True)


def plot(sensors, points, result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points.lon, points.lat, s=6, c="tab:gray", alpha=0.5, label="Behaviour points")
    ax.scatter(sensors.lon, sensors.lat, marker="s", s=50, c="tab:blue", label="Existing sensors")
    rec = result["recommended_sensors"]
    if rec:
        ax.scatter([r["lon"] for r in rec], [r["lat"] for r in rec], marker="*", s=250, c="tab:red", label="Recommended sensors")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    cov = result["coverage"]
    ax.set_title(f"Sensor coverage: {cov['before_pct']}% -> {cov['after_pct']}%")
    ax.legend(loc="upper right")
    ax.set_aspect(1 / np.cos(np.radians(CENTER[0])))
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/spatial/analyze")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=int, default=300, help="number of uniform background points")
    ap.add_argument("--eps", type=float, default=150.0)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--coverage-radius", type=float, default=200.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sensors, points = make_sensors(), make_points(rng, args.noise)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sensors.to_csv(out / "sensors.csv", index=False)
    points.to_csv(out / "points.csv", index=False)

    # Sanity check with haversine: distance from every point to its nearest sensor.
    d = np.min(haversine_m(points.lat.values[:, None], points.lon.values[:, None], sensors.lat.values[None, :], sensors.lon.values[None, :]), axis=1)
    print(f"{len(sensors)} sensors, {len(points)} points; median distance point->sensor {np.median(d):.0f} m")

    payload = {
        "sensors": sensors.to_dict("records"),
        "points": points[["lat", "lon"]].to_dict("records"),
        "eps_m": args.eps,
        "min_samples": args.min_samples,
        "coverage_radius_m": args.coverage_radius,
    }
    r = requests.post(args.url, json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    cov = result["coverage"]
    print(f"clusters: {len(result['clusters'])}, noise points: {result['noise_points']}")
    print(f"recommended new sensors: {len(result['recommended_sensors'])}")
    print(f"coverage: {cov['before_pct']}% -> {cov['after_pct']}%")

    if args.plot:
        plot(sensors, points, result, out / "map.png")
        print(f"saved {out / 'map.png'}")


if __name__ == "__main__":
    main()
