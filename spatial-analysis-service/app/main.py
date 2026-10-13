from fastapi import FastAPI

from . import analysis
from .schemas import AnalyzeRequest, AnalyzeResponse

# Aplikacija; naslov i opis se prikazuju u automatskoj dokumentaciji na /docs
app = FastAPI(
    title="spatial-analysis-service",
    description="Stateless service: finds hotspots in behaviour points (DBSCAN) and recommends sensor locations.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    # Jednostavna provera da li servis radi (koristi se i za Docker/gateway)
    return {"status": "UP"}


# POST /spatial/analyze: FastAPI sam pretvara JSON iz tela zahteva u AnalyzeRequest
# i proverava ga (loš unos -> 422), a odgovor oblikuje prema AnalyzeResponse.
@app.post("/spatial/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    # Ovo je samo tanka "ljuska": pretvara modele u obične n-torke i prepušta
    # ceo posao čistoj funkciji iz analysis.py. Servis ne pamti ništa (stateless).
    return analysis.analyze(
        sensors=[(s.id, s.lat, s.lon) for s in req.sensors],
        points=[(p.lat, p.lon) for p in req.points],
        eps_m=req.eps_m,
        min_samples=req.min_samples,
        coverage_radius_m=req.coverage_radius_m,
    )