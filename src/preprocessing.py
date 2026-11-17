import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # 1. Basic Cleaning
    # Remove cancelled/diverted flights for delay prediction
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
    
    # 2. Date/Time Features
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    
    # Extract Hour from CRS_DEP_TIME (HHMM format)
    df['HOUR_OF_DAY'] = (df['CRS_DEP_TIME'] // 100)
    
    # 3. Flight Features
    df['IS_SHORT_HAUL'] = (df['DISTANCE'] < 500).astype(int)
    
    # 4. Route Encoding (Basic)
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']
    
    # 5. PREV_FLIGHT_DELAYED (Proxy: Same carrier, same origin, previous hour)
    # This is a simplified cascade proxy as tail numbers are missing
    df = df.sort_values(['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'HOUR_OF_DAY'])
    df['PREV_FLIGHT_DELAYED'] = df.groupby(['FL_DATE', 'OP_CARRIER', 'ORIGIN'])['IS_DELAYED'].shift(1).fillna(0)
    
    # Select features for modeling
    features = [
        'MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND', 'HOUR_OF_DAY', 
        'DISTANCE', 'IS_SHORT_HAUL', 'OP_CARRIER', 'ORIGIN', 'DEST', 
        'PREV_FLIGHT_DELAYED'
    ]
    
    X = df[features]
    y = df['IS_DELAYED']
    
    # One-Hot Encoding for categorical variables
    X = pd.get_dummies(X, columns=['OP_CARRIER', 'ORIGIN', 'DEST'])
    
    return X, y, df

def chronological_split(X, y, test_size=0.2):
    """
    Splits data chronologically to respect temporal dependence.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y, df = preprocess_data("data/flights_2023_sample.csv")
    print(f"Preprocessed features shape: {X.shape}")
    print(f"Sample features: {X.columns[:10].tolist()}")
    
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
