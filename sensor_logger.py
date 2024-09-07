import time
import requests
from datetime import datetime
import random
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/emissions/")
API_KEY = os.getenv("API_KEY", "your_default_api_key")
SENSOR_ID = int(os.getenv("SENSOR_ID", "1"))
INDUSTRY = os.getenv("INDUSTRY", "electronic_manufacturing")

def read_sensor_data():
    # Simulating sensor readings
    if INDUSTRY == "electronic_manufacturing":
        return {
            "co2_emissions": random.uniform(80, 120),
            "energy_consumption": random.uniform(800, 1200),
            "production_volume": random.uniform(4000, 6000)
        }
    elif INDUSTRY == "oil_and_gas":
        return {
            "co2_emissions": random.uniform(800, 1200),
            "energy_consumption": random.uniform(8000, 12000),
            "production_volume": random.uniform(8000, 12000)
        }
    else:
        raise ValueError(f"Unsupported industry: {INDUSTRY}")

def send_reading_to_api(sensor_data):
    data = {
        "timestamp": datetime.now().isoformat(),
        "co2_emissions": sensor_data["co2_emissions"],
        "energy_consumption": sensor_data["energy_consumption"],
        "production_volume": sensor_data["production_volume"],
        "sensor_id": SENSOR_ID,
        "industry": INDUSTRY
    }
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Successfully sent reading: CO2 Emissions {data['co2_emissions']} units")
        else:
            print(f"Failed to send reading. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error sending data to API: {e}")
def main():
    print(f"Starting sensor logger for {INDUSTRY} industry")
    while True:
        sensor_data = read_sensor_data()
        send_reading_to_api(sensor_data)
        time.sleep(60)  # Wait for 1 minute before next reading

if __name__ == "__main__":
    main()