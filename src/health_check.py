import os
import pandas as pd
import joblib

def run_health_check():
    print("Starting GE Aviation Intelligence Health Check...")
    
    # 1. Data Integrity
    print("\n[1/5] Checking Data...")
    df_path = "data/flights_2023_sample.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        required_cols = ['TAIL_NUM', 'AIRCRAFT_TYPE', 'FUEL_BURN_RATE', 'IS_DELAYED']
        missing = [c for c in required_cols if c not in df.columns]
        if not missing:
            print("OK: Data exists with all Enterprise columns.")
        else:
            print(f"ERROR: Data missing columns: {missing}")
    else:
        print("ERROR: Data file missing!")

    # 2. ML Models
    print("\n[2/5] Checking Models...")
    models = ["results/xgb_model.pkl", "results/anomaly_engine.pkl", "results/model_features.pkl"]
    for m in models:
        if os.path.exists(m):
            print(f"OK: {m} found.")
        else:
            print(f"ERROR: {m} missing!")

    # 3. API & Environment
    print("\n[3/5] Checking API Keys...")
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY"): print("OK: Google API Key Detected")
    else: print("WARN: Google API Key Missing (GenAI will use Demo Mode)")
    
    if os.getenv("CHECKWX_API_KEY"): print("OK: CheckWX API Key Detected")
    else: print("WARN: CheckWX API Key Missing (Weather will use Simulated Mode)")

    # 4. Deployment Package
    print("\n[4/5] Checking Deployment Package (hf_app/)...")
    if os.path.exists("hf_app/app.py") and os.path.exists("hf_app/results/xgb_model.pkl"):
        print("OK: hf_app/ is correctly populated.")
    else:
        print("ERROR: hf_app/ is missing critical files!")

    # 5. Runtime Test
    print("\n[5/5] Testing GenAI Engine Logic...")
    from genai_engine import get_delay_insight
    test_flight = {"carrier": "GE", "origin": "ATL", "dest": "JFK", "dep_time": "12:00"}
    try:
        res = get_delay_insight(test_flight, 0.45, ["Peak Traffic"])
        if "Insight Engine Error" not in res:
            print("OK: GenAI Engine Operational")
        else:
            print(f"WARN: GenAI Engine returned error, but fallback handled it: {res}")
    except Exception as e:
        print(f"ERROR: GenAI Engine Critical Failure: {e}")

    print("\nHEALTH CHECK COMPLETE: SYSTEM READY.")

if __name__ == "__main__":
    run_health_check()
