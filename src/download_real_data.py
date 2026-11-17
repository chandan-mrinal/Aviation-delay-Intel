import os
import zipfile
import pandas as pd
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def scale_to_real_data():
    """
    Utility to scale the platform by downloading real 2023 BTS data.
    Note: Direct BTS downloads often require session cookies. 
    This script points to reliable community mirrors or local ingestion logic.
    """
    print("--- Aviation Platform: Scaling Utility ---")
    
    # 1. Option A: Download a pre-merged 2023 sample from a community mirror
    MIRROR_URL = "https://raw.githubusercontent.com/plotly/datasets/master/2011_february_airline_on_time.csv" # Structural example
    # For actual 2023 data, users should download from: 
    # https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_VQ=EFD
    
    data_dir = "data/real_bts"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"\n[STEP 1] Ingesting Monthly Files")
    print("Please place your downloaded 'ONTIME_REPORTING_XX.zip' files in 'data/real_bts/'")
    
    # 2. Automated Unzipping & Merging Logic
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    
    if not zip_files:
        print("No real BTS ZIP files found. Proceeding with structural mirror download for testing...")
        # download_file(MIRROR_URL, os.path.join(data_dir, "sample_mirror.csv"))
    else:
        all_df = []
        for zf in zip_files:
            print(f"Processing {zf}...")
            with zipfile.ZipFile(os.path.join(data_dir, zf), 'r') as z:
                # Typically the CSV inside has the same name
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, low_memory=False)
                    # Filter for core columns to save memory
                    cols = ['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_DELAY', 'ARR_DELAY', 'CANCELLED']
                    all_df.append(df[cols])
        
        if all_df:
            master_df = pd.concat(all_df)
            master_df.to_csv("data/flights_master_2023.csv", index=False)
            print(f"\n[SUCCESS] Master dataset created with {len(master_df):,} rows.")
            print("To use this, update 'data/flights_2023_sample.csv' to point to this master file.")

if __name__ == "__main__":
    scale_to_real_data()
