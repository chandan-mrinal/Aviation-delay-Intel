import pandas as pd
import numpy as np

def perform_cascade_analysis(df):
    """
    Identifies delay propagation sequences using carrier/airport/temporal proxies.
    Returns summary metrics for delay propagation.
    """
    # Ensure data is sorted by carrier, origin, and time
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df = df.sort_values(['OP_CARRIER', 'ORIGIN', 'FL_DATE', 'CRS_DEP_TIME'])
    
    # Identify if previous flight of same carrier at same origin was delayed
    # This represents 'Turnaround Delay' propagation
    df['IS_PREV_DELAYED'] = df.groupby(['OP_CARRIER', 'ORIGIN'])['IS_DELAYED'].shift(1).fillna(0)
    
    # Calculate Propagation Probabilities
    # Prob(Delay | Prev_Delay) vs Prob(Delay | Prev_No_Delay)
    total_prev_delayed = df[df['IS_PREV_DELAYED'] == 1]
    total_prev_not_delayed = df[df['IS_PREV_DELAYED'] == 0]
    
    prob_delay_given_prev = total_prev_delayed['IS_DELAYED'].mean()
    prob_delay_given_no_prev = total_prev_not_delayed['IS_DELAYED'].mean()
    
    propagation_lift = prob_delay_given_prev / prob_delay_given_no_prev if prob_delay_given_no_prev > 0 else 1
    
    # Calculate Cascade Lengths
    # A cascade is a sequence of consecutive delays for a carrier at an airport
    df['delay_group'] = (df['IS_DELAYED'] != df.groupby(['OP_CARRIER', 'ORIGIN'])['IS_DELAYED'].shift()).cumsum()
    cascade_lengths = df[df['IS_DELAYED'] == 1].groupby(['OP_CARRIER', 'ORIGIN', 'delay_group']).size()
    
    summary = {
        "avg_cascade_length": float(cascade_lengths.mean()),
        "max_cascade_length": int(cascade_lengths.max()),
        "propagation_lift": float(propagation_lift),
        "prob_delay_given_prev": float(prob_delay_given_prev),
        "prob_delay_given_no_prev": float(prob_delay_given_no_prev)
    }
    
    return summary, cascade_lengths

if __name__ == "__main__":
    df = pd.read_csv("data/flights_2023_sample.csv")
    df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)
    
    summary, cascades = perform_cascade_analysis(df)
    print("Cascade Analysis Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")
    
    # Save results
    import json
    with open("results/cascade_summary.json", "w") as f:
        json.dump(summary, f)
    print("\nResults saved to results/cascade_summary.json")
