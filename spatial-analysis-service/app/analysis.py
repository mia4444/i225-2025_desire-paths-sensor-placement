"""Ovaj fajl sadrži samo čiste funkcije za prostornu analizu podataka, odnosno
obrađuje geografske koordinate, pronalazi žarišta na mapi, proverava da li postojeći senzori
poktivaju ta područja i predlaže lokacije za postavljanje novih senzora"""
"""ZAŠTO je prostorna analiza razdvojena od Web-logike?
Ova projektantska praksa je utemeljena na programerskom principu Separation of Concerns, kao što su:
1) Separation of concerns
2) Easier testing - funkcije se mogu testirati nezavisno od Web aplikacije
3) Reusability - funkcije se mogu koristiti u različitim delovima sistema
4) Easier maintenance - izmene logike prostorne analize ne iziskuju nužno izmenu Web-logike
"""

#Cilj: utvrditi gde se problemi najčešće pojavljuju

"""Čiste funkcije prostorne analize (bez veb koda), pa se lako testiraju.

Rastojanja se računaju haversine formulom na sferi poluprečnika 6371 km.
Haversine formula služi za izračunavanje udaljenosti između dve tačke na površini sfere, 
na osnovu njihovih geografskih širina i dužina (latitude i longitude).
Prosečan poluprečnik Zemlje iznosi približno 6371 km.

Na nivou grada greška je daleko ispod 1 %.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np                       # rad sa nizovima brojeva
from sklearn.cluster import DBSCAN       # grupisanje tačaka po gustini
from sklearn.neighbors import BallTree   # prostorni indeks za brzo traženje najbližeg

# Poluprečnik Zemlje u metrima. Služi za pretvaranje radijana u metre i obrnuto.

EARTH_RADIUS_M = 6_371_000.0 #prosečan poluprečnik zemlje izražen u metrima


#Prva funkcija
def _rad(coords: Sequence[Sequence[float]]) -> np.ndarray:
    # Koordinate stižu kao [lat, lon] u stepenima, a scikit-learn haversine radi
    # isključivo u radijanima, pa ih parsiram (pretvaram iz stepeni u radijane).
    # reshape(-1, 2) garantuje oblik "n redova x 2 kolone", čak i za praznu listu.

    return np.radians(np.asarray(coords, dtype=float).reshape(-1, 2))
    #np.asarray() pretvara koordinate u numerički niz
    #reshape() organizuje podatke u redove sa po 2 koordinate
    #np.radians() pretvara stepene u radijane



def nearest_distances_m(points: Sequence[Sequence[float]], sensors: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Rastojanje u metrima od svake tačke do najbližeg senzora, i indeks tog senzora.

    Ako nema senzora, rastojanje je +beskonačno, a indeks -1.
    """
    n = len(points)
    if len(sensors) == 0:
        # Granični slučaj: bez senzora nijedna tačka nije pokrivena
        return np.full(n, np.inf), np.full(n, -1)
    # BallTree jednom izgradi stablo od svih senzora; svaka pretraga je zatim
    # brza (otprilike O(log m)) umesto da se prolazi kroz sve senzore.
    tree = BallTree(_rad(sensors), metric="haversine")
    # k=1 znači "daj mi samo jednog, najbližeg komšiju"
    dist, idx = tree.query(_rad(points), k=1)
    # dist je ugao u radijanima; množenjem poluprečnikom Zemlje dobijam metre
    return dist[:, 0] * EARTH_RADIUS_M, idx[:, 0]


def cluster_points(points: Sequence[Sequence[float]], eps_m: float, min_samples: int) -> np.ndarray:
    """DBSCAN oznake klastera sa haversine rastojanjem. Oznaka -1 znači šum.
    Šum= tačke koje ne pripadaju nijednom žarištu"""
    if len(points) == 0:
        return np.array([], dtype=int)
    # eps: radijus komšiluka. Zadajemo ga u metrima, a DBSCAN sa haversine
    # metrikom očekuje radijane, pa delimo poluprečnikom Zemlje.
    # min_samples: najmanji broj tačaka u komšiluku da bi tačka bila "jezgro" žarišta (težište klastera je prosek koordinata tačaka koje pripadaju žarištu).
    model = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    return model.fit_predict(_rad(points))


def coverage_pct(points: Sequence[Sequence[float]], sensors: Sequence[Sequence[float]], radius_m: float) -> float:
    """Procenat tačaka koje su na udaljenosti <= radius_m od nekog senzora."""
    if len(points) == 0:
        return 0.0
    dist, _ = nearest_distances_m(points, sensors)
    # (dist <= radius_m) daje niz True/False; prosek tog niza je udeo pokrivenih tačaka
    return round(float(np.mean(dist <= radius_m)) * 100.0, 2)


def analyze(
        sensors: List[Tuple[str, float, float]],
        points: List[Tuple[float, float]],
        eps_m: float,
        min_samples: int,
        coverage_radius_m: float,
) -> dict:
    """Glavna funkcija: nalazi žarišta, proverava pokrivenost i predlaže nove senzore."""
    sensor_ids = [s[0] for s in sensors]                # imena senzora ("S01", ...)
    sensor_xy = [(s[1], s[2]) for s in sensors]         # samo koordinate (lat, lon)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)

    # KORAK 1: grupiši tačke; labels[i] je broj žarišta tačke i, ili -1 za šum
    labels = cluster_points(pts, eps_m, min_samples)
    clusters, recommended = [], []

    # KORAK 2: obradi svako žarište posebno. Oznaku -1 (šum) izbacujemo iz skupa.
    for label in sorted(set(labels.tolist()) - {-1}):
        members = pts[labels == label]                  # sve tačke ovog žarišta
        # Težište = prosek geografskih širina i dužina. Na razmeri grada
        # (nekoliko km) ovo je dovoljno tačno i mnogo jednostavnije od sferne geometrije.
        centroid = members.mean(axis=0)

        # KORAK 3: koliko je težište daleko od najbližeg postojećeg senzora?
        dist, idx = nearest_distances_m([centroid], sensor_xy)
        d: Optional[float] = None if np.isinf(dist[0]) else round(float(dist[0]), 1)
        sid: Optional[str] = sensor_ids[int(idx[0])] if idx[0] >= 0 else None

        # Žarište je "pokriveno" ako postoji senzor unutar radijusa pokrivenosti
        covered = d is not None and d <= coverage_radius_m
        clusters.append(
            {
                "cluster_id": int(label),
                "size": int(len(members)),
                "lat": round(float(centroid[0]), 6),
                "lon": round(float(centroid[1]), 6),
                "nearest_sensor_id": sid,
                "nearest_sensor_distance_m": d,
                "covered": bool(covered),
            }
        )

        # KORAK 4: nepokriveno žarište je "senzorska pustinja", pa predlažemo
        # novi senzor tačno u njegovom težištu
        if not covered:
            recommended.append(
                {
                    "lat": round(float(centroid[0]), 6),
                    "lon": round(float(centroid[1]), 6),
                    "supporting_points": int(len(members)),   # koliko tačaka podržava predlog
                    "cluster_id": int(label),
                }
            )

    # Najpre najveća žarišta: ona donose najviše koristi
    recommended.sort(key=lambda r: r["supporting_points"], reverse=True)

    # KORAK 5: pokrivenost pre i posle dodavanja predloženih senzora.
    # Računa se nad SVIM tačkama, uključujući šum, pa "posle" retko dostigne 100 %.
    before = coverage_pct(pts, sensor_xy, coverage_radius_m)
    after = coverage_pct(pts, sensor_xy + [(r["lat"], r["lon"]) for r in recommended], coverage_radius_m)

    return {
        "total_points": int(len(pts)),
        "noise_points": int(np.sum(labels == -1)),
        "clusters": clusters,
        "recommended_sensors": recommended,
        "coverage": {"before_pct": before, "after_pct": after},
    }