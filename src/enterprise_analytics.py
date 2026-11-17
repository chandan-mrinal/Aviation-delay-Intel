import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import json
import os

def run_sustainability_analysis(df):
    """
    Calculates fuel penalty and CO2 emissions for delayed flights.
    GE Engine Focus: CO2 = Fuel (kg) * 3.16
    """
    # Only calculate for delayed flights
    delayed = df[df['DEP_DELAY'] > 0].copy()
    
    # Fuel penalty during ground idle/congestion (assuming 15% of hourly burn rate per hour of delay)
    delayed['FUEL_PENALTY_KG'] = (delayed['DEP_DELAY'] / 60.0) * delayed['FUEL_BURN_RATE'] * 0.15
    delayed['CO2_PENALTY_KG'] = delayed['FUEL_PENALTY_KG'] * 3.16
    
    summary = {
        "total_fuel_wasted_kg": float(delayed['FUEL_PENALTY_KG'].sum()),
        "total_co2_penalty_kg": float(delayed['CO2_PENALTY_KG'].sum()),
        "avg_co2_per_delay": float(delayed['CO2_PENALTY_KG'].mean())
    }
    
    return summary

def train_anomaly_engine(df):
    """
    Uses Isolation Forest to detect Black Swan events.
    Features: DEP_DELAY, DISTANCE, HOUR_OF_DAY
    """
    df_clean = df.copy()
    df_clean['HOUR'] = df_clean['CRS_DEP_TIME'] // 100
    
    features = ['DEP_DELAY', 'DISTANCE', 'HOUR']
    X = df_clean[features].fillna(0)
    
    # contamination=0.01 (look for top 1% outliers)
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X)
    
    # Save the model
    os.makedirs("results", exist_ok=True)
    joblib.dump(iso_forest, "results/anomaly_engine.pkl")
    
    # Return count of anomalies found in training data
    preds = iso_forest.predict(X)
    return int((preds == -1).sum())

def analyze_tail_rotation(df):
    """
    Identifies the 'Most Tired' airframes (most accumulated delay).
    """
    tail_stats = df.groupby('TAIL_NUM').agg({
        'DEP_DELAY': ['sum', 'count', 'mean'],
        'AIRCRAFT_TYPE': 'first'
    }).reset_index()
    
    tail_stats.columns = ['TAIL_NUM', 'TOTAL_DELAY', 'FLIGHT_COUNT', 'AVG_DELAY', 'AIRCRAFT_TYPE']
    return tail_stats.sort_values('TOTAL_DELAY', ascending=False).head(20)

if __name__ == "__main__":
    print("Running Enterprise Analytics...")
    df = pd.read_csv("data/flights_2023_sample.csv")
    
    # Sustainability
    sus_summary = run_sustainability_analysis(df)
    with open("results/sustainability_summary.json", "w") as f:
        json.dump(sus_summary, f)
    print(f"Sustainability Analysis Complete: {sus_summary['total_co2_penalty_kg']/1000:.1f} tons CO2 penalty.")
    
    # Anomaly
    num_anomalies = train_anomaly_engine(df)
    print(f"Anomaly Engine Trained. Detected {num_anomalies} training outliers.")
    
    # Tail Rotation
    top_tails = analyze_tail_rotation(df)
    top_tails.to_csv("results/tail_rotation_stats.csv", index=False)
    print("Tail Rotation Analysis Complete.")
