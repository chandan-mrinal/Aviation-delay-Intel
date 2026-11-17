import requests
import os
import numpy as np

def get_airport_weather(icao_code, month):
    """
    Fetches real METAR data if API_KEY is present, otherwise falls back to simulation.
    """
    api_key = os.getenv("CHECKWX_API_KEY")
    
    if api_key:
        try:
            url = f"https://api.checkwx.com/metar/{icao_code}/decoded"
            headers = {"X-API-Key": api_key}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            if data.get('data'):
                metar = data['data'][0]
                return {
                    "visibility": metar.get('visibility', {}).get('miles', 10.0),
                    "wind_speed": metar.get('wind', {}).get('speed_kts', 5),
                    "precip_type": "Rain" if metar.get('rain') else "None",
                    "temp": metar.get('temperature', {}).get('celsius', 20),
                    "is_live": True
                }
        except Exception as e:
            print(f"Weather API Error: {e}")

    # Simulation Fallback (Existing Logic)
    weather = {
        "visibility": 10.0,
        "wind_speed": 5,
        "precip_type": "None",
        "temp": 20,
        "is_live": False
    }
    
    # Regional Logic
    if icao_code in ['ORD', 'DEN', 'BOS', 'EWR', 'JFK']:
        if month in [12, 1, 2]: # Winter
            weather['precip_type'] = np.random.choice(["None", "Snow", "Sleet"], p=[0.7, 0.2, 0.1])
            weather['temp'] = np.random.randint(-15, 5)
            weather['wind_speed'] = np.random.randint(10, 30)
            if weather['precip_type'] != "None":
                weather['visibility'] = np.random.uniform(0.5, 3.0)
                
    elif icao_code in ['ATL', 'MIA', 'MCO', 'DFW']:
        if month in [6, 7, 8]: # Summer Storms
            weather['precip_type'] = np.random.choice(["None", "Rain", "TSRA"], p=[0.6, 0.3, 0.1])
            weather['temp'] = np.random.randint(25, 35)
            if weather['precip_type'] == "TSRA":
                weather['wind_speed'] = np.random.randint(20, 45)
                weather['visibility'] = np.random.uniform(1.0, 5.0)
                
    elif icao_code == 'SFO':
        # Fog Logic
        weather['precip_type'] = np.random.choice(["None", "Fog"], p=[0.6, 0.4])
        if weather['precip_type'] == "Fog":
            weather['visibility'] = np.random.uniform(0.1, 1.0)
            
    return weather

def format_metar(icao, weather):
    """Formats simulated weather into a readable METAR string."""
    precip = f" {weather['precip_type']}" if weather['precip_type'] != "None" else ""
    return f"{icao} 261053Z {weather['wind_speed']:02d}KT {weather['visibility']:.1f}SM{precip} {weather['temp']}/{weather['temp']-2} A2992"

if __name__ == "__main__":
    for apt in ['ORD', 'MIA', 'SFO', 'ATL']:
        w = get_airport_weather(apt, 1)
        print(format_metar(apt, w))
