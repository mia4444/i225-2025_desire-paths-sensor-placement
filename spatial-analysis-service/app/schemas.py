#Ugovor servisa: Kako će JSON izgledati na ulazu i izlazu
from typing import List, Optional
from pydantic import BaseModel, Field #Istovremeno i dokumentacija i validator (pydantic klase služe za ispunjavanje te 2 uloge simultano)
#Pydantic klase nasleđuju od pydantic.BaseModel te ih možemo nazvati i Modeli
#Omogućavaju automatizovani runtime validation, type coercion i parsiranje
#Pydantic actively enforces data types at runtime
class Sensor (BaseModel): #Sensor je klasa koja nasleđuje BaseModel i samim tim je pydantic model. Senzor je vrsta BaseModel-a.
    id: str
    lat: float= Field(ge=-90, le=90) #automatski odbija nemoguće koordinate, javlja se greška 422 kao u Java servisima
    lon: float=Field(ge=-180, le=180)
    #ge= greater than or equal
    #le= less than or equal

class Point (BaseModel):
    """Ovo je jedno zabeleženo ponašanje koje je reprezentovano tačkom"""
    #Uzorak "desire path"-a ili prijava incidenta
    lat: float= Field(ge=-90, le=90)
    lon: float=Field(ge=-180, le=180)

class AnalyzeRequest(BaseModel):
    #Senzori mogu biti izostavljeni: prazna lista
    sensors: List [Sensor]=Field(default_factory=list)
    points: List[Point]=Field(max_length=10_000) #Max 10.000 tačaka: potrošnja memorije DBSCAN-a raste približno kvadratno
    #Dakle, max_length štiti server od prevelikih zahteva
    eps_m:float=Field(default=150.0,gt=0,decription="DBSCAN radijus komšiluka u metrima")
    min_samples:int=Field(default=5,ge=1,description="DBSCAN najmanji broj tačaka u žarištu/klasteru")
    coverage_radius_m: float = Field(default=200.0, gt=0, description="Distance within which a sensor 'covers' a point")

#Ispod su sve klase odgovora: FastAPI ih koristi da formira JSON output
class Cluster(BaseModel):
    cluster_id: int
    size: int
    lat: float #težište žarišta/klastera
    lon: float
    nearest_sensor_id: Optional[str] #None ako uopšte nema senzora
    nearest_sensor_distance_m: Optional[float]
    covered: bool #da li je žarište pokriveno postojećim senzorom


class RecommendedSensor(BaseModel):
    lat: float
    lon: float
    supporting_points: int
    cluster_id: int


class Coverage(BaseModel):
    before_pct: float
    after_pct: float


class AnalyzeResponse(BaseModel):
    total_points: int
    noise_points: int
    clusters: List[Cluster]
    recommended_sensors: List[RecommendedSensor]
    coverage: Coverage
