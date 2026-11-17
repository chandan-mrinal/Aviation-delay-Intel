import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=100000):
    """
    Generates a high-fidelity synthetic dataset following the BTS On-Time Performance schema.
    This is used for development and demonstration purposes when the full multi-GB dataset is not available.
    """
    np.random.seed(42)
    
    # Constants
    carriers = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9']
    airports = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS', 'PHX', 'MIA', 'SEA', 'EWR', 'SFO', 'JFK', 'BOS']
    
    # Generate dates for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_rows)]
    
    data = {
        'FL_DATE': [d.strftime('%Y-%m-%d') for d in dates],
        'OP_CARRIER': np.random.choice(carriers, num_rows),
        'ORIGIN': np.random.choice(airports, num_rows),
        'DEST': np.random.choice(airports, num_rows),
        'CRS_DEP_TIME': np.random.randint(0, 2400, num_rows),
        'CRS_ARR_TIME': np.random.randint(0, 2400, num_rows),
        'DISTANCE': np.random.randint(100, 3000, num_rows),
    }
    
    # Ensure ORIGIN != DEST
    for i in range(num_rows):
        while data['ORIGIN'][i] == data['DEST'][i]:
            data['DEST'][i] = np.random.choice(airports)
            
    # Enterprise Features: Tail Tracking & Aircraft Types
    tail_numbers = [f"N{np.random.randint(100, 999)}GE" for _ in range(500)]
    aircraft_types = ["Boeing 737-800", "Airbus A320neo", "Boeing 787-9", "Embraer E195"]
    fuel_rates = {"Boeing 737-800": 2500, "Airbus A320neo": 2200, "Boeing 787-9": 5400, "Embraer E195": 1800} # kg/hr
    
    data['TAIL_NUM'] = [np.random.choice(tail_numbers) for _ in range(num_rows)]
    data['AIRCRAFT_TYPE'] = [np.random.choice(aircraft_types) for _ in range(num_rows)]
    data['FUEL_BURN_RATE'] = [fuel_rates[t] for t in data['AIRCRAFT_TYPE']]
    
    # Generate Delays with stronger hourly signals
    df_temp = pd.DataFrame({'HOUR': data['CRS_DEP_TIME'] // 100})
    hourly_prob = 0.1 + (df_temp['HOUR'] / 23.0) * 0.5
    
    data['DEP_DELAY'] = np.random.normal(5, 10, num_rows)
    data['ARR_DELAY'] = data['DEP_DELAY'] + np.random.normal(0, 5, num_rows)
    
    # Apply significant delays
    random_vals = np.random.rand(num_rows)
    heavy_delay_mask = random_vals < hourly_prob
    
    data['DEP_DELAY'][heavy_delay_mask] += np.random.exponential(45, heavy_delay_mask.sum())
    data['ARR_DELAY'][heavy_delay_mask] += np.random.exponential(50, heavy_delay_mask.sum())
    
    # INJECT ANOMALIES (Phase 5 Feature 4)
    # 1% of flights have "impossible" delays or weird patterns for the Anomaly Engine to find
    anomaly_idx = np.random.choice(num_rows, int(num_rows * 0.01), replace=False)
    data['DEP_DELAY'][anomaly_idx] += 500 # 8-hour delays
    data['DISTANCE'][anomaly_idx] = 50 # Short flights with massive delays
    
    # Rounding
    data['DEP_DELAY'] = np.round(data['DEP_DELAY'])
    data['ARR_DELAY'] = np.round(data['ARR_DELAY'])
    
    # Cancellations and Diversions
    data['CANCELLED'] = np.random.choice([0, 1], num_rows, p=[0.98, 0.02])
    data['DIVERTED'] = np.random.choice([0, 1], num_rows, p=[0.995, 0.005])
    
    # Delay Causes (only for ARR_DELAY > 15)
    data['CARRIER_DELAY'] = 0.0
    data['WEATHER_DELAY'] = 0.0
    data['NAS_DELAY'] = 0.0
    data['SECURITY_DELAY'] = 0.0
    data['LATE_AIRCRAFT_DELAY'] = 0.0
    
    delayed_mask = (data['ARR_DELAY'] > 15) & (data['CANCELLED'] == 0)
    num_delayed = delayed_mask.sum()
    
    if num_delayed > 0:
        # Distribute delay across causes
        shares = np.random.dirichlet([2, 1, 3, 0.1, 4], num_delayed)
        delayed_arr_delay = np.array(data['ARR_DELAY'])[delayed_mask]
        
        carrier_delay = np.zeros(num_rows)
        weather_delay = np.zeros(num_rows)
        nas_delay = np.zeros(num_rows)
        security_delay = np.zeros(num_rows)
        late_aircraft_delay = np.zeros(num_rows)
        
        carrier_delay[delayed_mask] = shares[:, 0] * delayed_arr_delay
        weather_delay[delayed_mask] = shares[:, 1] * delayed_arr_delay
        nas_delay[delayed_mask] = shares[:, 2] * delayed_arr_delay
        security_delay[delayed_mask] = shares[:, 3] * delayed_arr_delay
        late_aircraft_delay[delayed_mask] = shares[:, 4] * delayed_arr_delay
        
        data['CARRIER_DELAY'] = carrier_delay
        data['WEATHER_DELAY'] = weather_delay
        data['NAS_DELAY'] = nas_delay
        data['SECURITY_DELAY'] = security_delay
        data['LATE_AIRCRAFT_DELAY'] = late_aircraft_delay

    df = pd.DataFrame(data)
    
    # Target variable
    df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic aviation data...")
    df = generate_synthetic_data(100000)
    df.to_csv("data/flights_2023_sample.csv", index=False)
    print(f"Dataset generated with {len(df)} rows.")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Delay rate: {df['IS_DELAYED'].mean():.2%}")
