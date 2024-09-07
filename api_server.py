from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
import databases
import sqlalchemy
from sqlalchemy import Table, Column, Integer, Float, DateTime, String, MetaData
import os
from contextlib import asynccontextmanager
import json
from sqlalchemy.ext.declarative import declarative_base

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./co2emissions.db")
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

Base = declarative_base()

emissions = Table(
    "emissions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime),
    Column("co2_emissions", Float),
    Column("energy_consumption", Float),
    Column("production_volume", Float),
    Column("sensor_id", Integer),
    Column("industry", String),
    Column("equipment", String)
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
    equipment: str

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

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>CO2 Emissions API</title>
            <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f0f0f0;
                }
                .container {
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                textarea {
                    width: 100%;
                    height: 150px;
                }
                button {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 15px 32px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                }
                #result {
                    margin-top: 20px;
                    white-space: pre-wrap;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    table-layout: fixed;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .modal {
                    display: none;
                    position: fixed;
                    z-index: 1;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.4);
                }
                .modal-content {
                    background-color: #fefefe;
                    margin: 15% auto;
                    padding: 20px;
                    border: 1px solid #888;
                    width: 80%;
                    max-width: 500px;
                }
                .close {
                    color: #aaa;
                    float: right;
                    font-size: 28px;
                    font-weight: bold;
                }
                .close:hover,
                .close:focus {
                    color: black;
                    text-decoration: none;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to the CO2 Emissions API by Bright Bassey</h1>
                <h2>Upload Emissions Data</h2>
                <textarea id="jsonData" placeholder="Paste your JSON data here"></textarea>
                <button onclick="uploadData()">Upload Data</button>
                <h2>Retrieve Emissions Data</h2>
                <button onclick="getEmissions()">Get Emissions</button>
                <div id="result"></div>
            </div>
            <div id="successModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <h2>Success</h2>
                    <p id="successMessage"></p>
                </div>
            </div>
            <div id="errorModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <h2>Error</h2>
                    <p id="errorMessage"></p>
                </div>
            </div>
            <script>
                const apiKey = 'your_default_api_key';  // Replace with your actual API key
                
                async function uploadData() {
                    const jsonData = document.getElementById('jsonData').value;
                    try {
                        const response = await axios.post('/emissions/', JSON.parse(jsonData), {
                            headers: {
                                'X-API-Key': apiKey,
                                'Content-Type': 'application/json'
                            }
                        });
                        showModal('successModal', 'Upload successful: ' + JSON.stringify(response.data, null, 2));
                    } catch (error) {
                        showModal('errorModal', 'Error: ' + (error.response ? error.response.data.detail : error.message));
                    }
                }
                
                async function getEmissions() {
                    try {
                        const response = await axios.get('/emissions/', {
                            headers: {
                                'X-API-Key': apiKey
                            }
                        });
                        const emissions = response.data;
                        const table = createEmissionsTable(emissions);
                        document.getElementById('result').innerHTML = '';
                        document.getElementById('result').appendChild(table);
                    } catch (error) {
                        showModal('errorModal', 'Error: ' + (error.response ? error.response.data.detail : error.message));
                    }
                }

                function createEmissionsTable(emissions) {
                    const table = document.createElement('table');
                    const headers = ['Timestamp', 'CO2 Emissions', 'Energy Consumption', 'Production Volume', 'Sensor ID', 'Industry', 'Equipment'];
                    
                    // Create table header
                    const headerRow = table.insertRow();
                    headers.forEach(header => {
                        const th = document.createElement('th');
                        th.textContent = header;
                        headerRow.appendChild(th);
                    });

                    // Create table rows
                    emissions.forEach(emission => {
                        const row = table.insertRow();
                        row.insertCell().textContent = emission.timestamp;
                        row.insertCell().textContent = emission.co2_emissions;
                        row.insertCell().textContent = emission.energy_consumption;
                        row.insertCell().textContent = emission.production_volume;
                        row.insertCell().textContent = emission.sensor_id;
                        row.insertCell().textContent = emission.industry;
                        row.insertCell().textContent = emission.equipment;
                    });

                    return table;
                }

                function showModal(modalId, message) {
                    const modal = document.getElementById(modalId);
                    const span = modal.getElementsByClassName("close")[0];
                    const messageElement = modal.getElementsByTagName('p')[0];
                    
                    messageElement.innerText = message;
                    modal.style.display = "block";
                    
                    span.onclick = function() {
                        modal.style.display = "none";
                    }
                    
                    window.onclick = function(event) {
                        if (event.target == modal) {
                            modal.style.display = "none";
                        }
                    }
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)