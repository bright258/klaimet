from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime
import databases
import sqlalchemy
from sqlalchemy import Table, Column, Integer, Float, DateTime, String, MetaData
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./co2emissions.db")
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

emissions = Table(
    "emissions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime),
    Column("co2_emissions", Float),
    Column("energy_consumption", Float),
    Column("production_volume", Float),
    Column("sensor_id", Integer),
    Column("industry", String)
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

app = FastAPI()

# Security
API_KEY = os.getenv("API_KEY", "your_default_api_key")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key

# Pydantic models
class EmissionReading(BaseModel):
    timestamp: datetime
    co2_emissions: float
    energy_consumption: float
    production_volume: float
    sensor_id: int
    industry: str

class EmissionResponse(EmissionReading):
    id: int

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/emissions/", response_model=EmissionResponse, dependencies=[Depends(get_api_key)])
async def create_emission(emission: EmissionReading):
    query = emissions.insert().values(**emission.dict())
    last_record_id = await database.execute(query)
    return {**emission.dict(), "id": last_record_id}

@app.get("/emissions/", response_model=list[EmissionResponse], dependencies=[Depends(get_api_key)])
async def read_emissions(start_date: datetime = None, end_date: datetime = None, industry: str = None):
    query = emissions.select()
    if start_date:
        query = query.where(emissions.c.timestamp >= start_date)
    if end_date:
        query = query.where(emissions.c.timestamp <= end_date)
    if industry:
        query = query.where(emissions.c.industry == industry)
    return await database.fetch_all(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)