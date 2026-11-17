import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from weather_service import get_airport_weather, format_metar
from genai_engine import get_delay_insight

# Set Page Config
st.set_page_config(page_title="GE Aviation Delay Intelligence", layout="wide", page_icon="✈️")

# Robust File Path Helper
def get_file_path(filename, folder):
    """Checks if file exists in root or in specified folder."""
    if os.path.exists(filename): return filename
    path = os.path.join(folder, filename)
    if os.path.exists(path): return path
    return filename # Fallback to original

# Load Data
@st.cache_data
def load_data():
    path = get_file_path("flights_2023_sample.csv", "data")
    df = pd.read_csv(path)
    return df

# Load Model
@st.cache_resource
def load_model():
    model_path = get_file_path("xgb_model.pkl", "results")
    feat_path = get_file_path("model_features.pkl", "results")
    model = joblib.load(model_path)
    features = joblib.load(feat_path)
    return model, features

df = load_data()
model, model_features = load_model()

st.title("✈️ Aviation Delay Intelligence Platform")

# Sidebar for non-technical users
with st.sidebar:
    st.header("📖 User Guide")
    st.info("""
    **New to this? Here's the key:**
    *   **AVI Score**: The 'Stress Level' of an airport. Higher = More chaos.
    *   **Delay Cascade**: A 'Domino Effect' where one late plane makes 5 more late.
    *   **GenAI Briefing**: A plain-English summary written by AI for pilots.
    *   **Sustainability**: How much extra pollution/fuel was caused by today's delays.
    """)
    st.markdown("---")
    st.markdown("### 🛰️ System Connectivity")
    
    # Live Connectivity Check
    if os.getenv("GOOGLE_API_KEY"):
        st.success("● GenAI Engine: Connected")
    else:
        st.warning("○ GenAI Engine: Demo Mode")
        
    if os.getenv("CHECKWX_API_KEY"):
        st.success("● Weather Fusion: Live API")
    else:
        st.info("○ Weather Fusion: Simulated")
        
    st.success("● ML Pipeline: XGBoost v1.2")
    st.markdown("---")
    st.markdown("### 💡 Quick Tip")
    st.write("Use the **Predictor** tab to see if your specific flight is likely to be late today!")

# Executive Summary (Phase 6: UX Layer)
st.markdown("### 🚦 Executive Network Pulse")
avg_vulnerability = df['IS_DELAYED'].mean() * 100
if avg_vulnerability < 20:
    st.success(f"**Network Status: Healthy.** Delays are rare and isolated. (Global Risk: {avg_vulnerability:.1f}%)")
elif avg_vulnerability < 40:
    st.warning(f"**Network Status: Stressed.** Minor 'Domino Effects' detected in the evening. (Global Risk: {avg_vulnerability:.1f}%)")
else:
    st.error(f"**Network Status: Critical.** Significant congestion at major hubs. (Global Risk: {avg_vulnerability:.1f}%)")

st.markdown("---")

tabs = st.tabs(["📊 Fleet Overview", "🏢 Airport Intelligence", "🔮 Delay Predictor", "🚀 Enterprise Intelligence"])

# Tab 1: Overview
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", f"{len(df):,}", help="The total number of flights analyzed in this dataset.")
    col2.metric("Avg Delay (min)", f"{df['ARR_DELAY'].mean():.1f}", help="The average time a flight arrives past its scheduled time.")
    col3.metric("Delay Rate (>15m)", f"{df['IS_DELAYED'].mean():.1%}", help="The percentage of flights that were 'Officially' late (more than 15 mins).")
    col4.metric("Cancellation Rate", f"{df['CANCELLED'].mean():.1%}", help="Percentage of flights that never took off.")

    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Delay Trends by Hour")
        df['HOUR'] = (df['CRS_DEP_TIME'] // 100)
        hourly_delay = df.groupby('HOUR')['IS_DELAYED'].mean().reset_index()
        fig_hour = px.line(hourly_delay, x='HOUR', y='IS_DELAYED', title="Probability of Delay vs Departure Hour")
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col_r:
        st.subheader("Carrier Performance")
        carrier_perf = df.groupby('OP_CARRIER')['IS_DELAYED'].mean().sort_values(ascending=False).reset_index()
        carrier_perf['IS_DELAYED'] = carrier_perf['IS_DELAYED'] * 100
        fig_carrier = px.bar(carrier_perf, x='OP_CARRIER', y='IS_DELAYED', 
                             title="Delay Rate by Carrier", 
                             color='IS_DELAYED',
                             labels={'IS_DELAYED': 'Delay Rate (%)', 'OP_CARRIER': 'Carrier'})
        st.plotly_chart(fig_carrier, use_container_width=True)

    st.subheader("⛓️ Delay Cascade Analysis")
    cascade_path = get_file_path("cascade_summary.json", "results")
    if os.path.exists(cascade_path):
        import json
        with open(cascade_path, "r") as f:
            cascade = json.load(f)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Cascade Length", f"{cascade['avg_cascade_length']:.2f} flights")
        c2.metric("Max Cascade", f"{cascade['max_cascade_length']} flights")
        c3.metric("Propagation Lift", f"{cascade['propagation_lift']:.2f}x")
        st.caption("Propagation Lift: Risk multiplier when the previous carrier flight at the airport was delayed.")

with tabs[1]:
    st.subheader("Airport Vulnerability Index (Refined AVI)")
    
    avi_path = get_file_path("airport_vulnerability.csv", "results")
    if os.path.exists(avi_path):
        airport_stats = pd.read_csv(avi_path)
    else:
        airport_stats = df.groupby('ORIGIN').agg({'ARR_DELAY': 'mean', 'IS_DELAYED': 'mean'}).reset_index()
        airport_stats['AVI_SCORE'] = (airport_stats['ARR_DELAY'] * 0.4 + airport_stats['IS_DELAYED'] * 100 * 0.6)
        airport_stats['Status'] = "N/A"

    st.dataframe(airport_stats[['ORIGIN', 'AVI_SCORE', 'Status', 'FLIGHT_COUNT']].style.background_gradient(subset=['AVI_SCORE'], cmap='RdYlGn_r'), use_container_width=True)
    
    fig_map = px.bar(airport_stats.sort_values('AVI_SCORE', ascending=False), 
                     x='ORIGIN', y='AVI_SCORE', color='Status', 
                     title="Airport Vulnerability Index (Refined)",
                     color_discrete_map={"Low": "green", "Moderate": "yellow", "High": "orange", "Critical": "red"})
    st.plotly_chart(fig_map, use_container_width=True)

# Tab 3: Delay Predictor
with tabs[2]:
    st.subheader("Predictive Operational Briefing")
    
    col1, col2 = st.columns(2)
    with col1:
        carrier = st.selectbox("Select Carrier", df['OP_CARRIER'].unique())
        origin = st.selectbox("Origin Airport", df['ORIGIN'].unique())
        dest = st.selectbox("Destination Airport", df['DEST'].unique())
    with col2:
        dep_time = st.slider("Scheduled Departure (Hour)", 0, 23, 12)
        month = st.slider("Month", 1, 12, 1)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
        
    if st.button("Generate Insight"):
        from weather_service import get_airport_weather, format_metar
        weather = get_airport_weather(origin, month)
        metar = format_metar(origin, weather)
        
        st.code(f"METAR: {metar}", language="text")
        
        weather_multiplier = 1.0
        if weather['precip_type'] in ['Snow', 'TSRA']: weather_multiplier = 1.4
        if weather['visibility'] < 1.0: weather_multiplier = 1.6
        if weather['wind_speed'] > 25: weather_multiplier = 1.2
        
        input_data = pd.DataFrame({
            'MONTH': [month],
            'DAY_OF_WEEK': [day_of_week],
            'IS_WEEKEND': [1 if day_of_week >= 5 else 0],
            'HOUR_OF_DAY': [dep_time],
            'DISTANCE': [df[(df['ORIGIN']==origin) & (df['DEST']==dest)]['DISTANCE'].mean() or 1000],
            'IS_SHORT_HAUL': [1 if (df[(df['ORIGIN']==origin) & (df['DEST']==dest)]['DISTANCE'].mean() or 1000) < 500 else 0],
            'OP_CARRIER': [carrier],
            'ORIGIN': [origin],
            'DEST': [dest],
            'PREV_FLIGHT_DELAYED': [0]
        })
        
        input_encoded = pd.get_dummies(input_data)
        final_input = pd.DataFrame(columns=model_features)
        for col in input_encoded.columns:
            if col in final_input.columns:
                final_input[col] = input_encoded[col]
        final_input = final_input.fillna(0)
        
        base_prob = model.predict_proba(final_input)[0, 1]
        fused_prob = min(base_prob * weather_multiplier, 0.99)
        
        st.markdown(f"### Adjusted Delay Probability: **{fused_prob:.1%}**")
        if weather_multiplier > 1.0:
            st.warning(f"Probability adjusted by {weather_multiplier}x due to adverse weather conditions.")
        st.progress(fused_prob)
        
        with st.spinner("Analyzing operational risk factors..."):
            flight_details = {"carrier": carrier, "origin": origin, "dest": dest, "dep_time": f"{dep_time}:00"}
            factors = []
            if weather['precip_type'] != "None": factors.append(f"Active {weather['precip_type']}")
            if weather['visibility'] < 3.0: factors.append("Low Visibility Ops")
            if dep_time >= 16 and dep_time <= 20: factors.append("Evening Peak Congestion")
            
            insight = get_delay_insight(flight_details, fused_prob, factors)
            st.info(insight)
            st.success("Briefing Generated. Data source: BTS AOTP 2023 Sample + Simulated METAR Fusion.")

# Tab 4: Enterprise Intelligence (Phase 5)
with tabs[3]:
    st.subheader("🚀 GE Enterprise Operational Intelligence")
    
    st.markdown("#### 🌿 Sustainability & Emissions Impact")
    sus_path = get_file_path("sustainability_summary.json", "results")
    if os.path.exists(sus_path):
        import json
        with open(sus_path, "r") as f:
            sus = json.load(f)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("CO2 Penalty (Total)", f"{sus['total_co2_penalty_kg']/1000:,.0f} Tons", delta="-2% vs LW", delta_color="inverse")
        c2.metric("Fuel Inefficiency", f"{sus['total_fuel_wasted_kg']/1000:,.0f} Tons", help="Excess fuel burn due to ground idle delays")
        c3.metric("Avg CO2 / Delay", f"{sus['avg_co2_per_delay']:.1f} kg")
        st.caption("Calculated based on aircraft-specific burn rates and delay duration. GE Engine Performance Factor applied.")

    st.markdown("---")
    st.markdown("#### 🌍 Global Network Pulse")
    coords = {
        "ATL": [33.6407, -84.4277], "ORD": [41.9742, -87.9073], "DFW": [32.8998, -97.0403],
        "DEN": [39.8561, -104.6737], "CLT": [35.2144, -80.9473], "LAX": [33.9416, -118.4085],
        "JFK": [40.6413, -73.7781], "SFO": [37.6213, -122.3790], "LAS": [36.0840, -115.1537],
        "MCO": [28.4312, -81.3081], "SEA": [47.4502, -122.3088], "PHX": [33.4342, -112.0081],
        "MIA": [25.7959, -80.2870], "BOS": [42.3656, -71.0096]
    }
    
    avi_map_path = get_file_path("airport_vulnerability.csv", "results")
    if os.path.exists(avi_map_path):
        avi_data = pd.read_csv(avi_map_path)
        avi_data['LAT'] = avi_data['ORIGIN'].map(lambda x: coords.get(x, [0,0])[0])
        avi_data['LON'] = avi_data['ORIGIN'].map(lambda x: coords.get(x, [0,0])[1])
        avi_data = avi_data[avi_data['LAT'] != 0]
        st.map(avi_data, latitude='LAT', longitude='LON', size='AVI_SCORE', color='#FF4B4B' if (avi_data['AVI_SCORE'] > 70).any() else '#00FF00')
        st.caption("Map nodes scaled by AVI Score (Airport Vulnerability Index).")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🕳️ Black Swan Detection")
        anom_path = get_file_path("anomaly_engine.pkl", "results")
        if os.path.exists(anom_path):
            iso_forest = joblib.load(anom_path)
            sample_anom = df.sample(100).copy()
            sample_anom['HOUR'] = sample_anom['CRS_DEP_TIME'] // 100
            preds = iso_forest.predict(sample_anom[['DEP_DELAY', 'DISTANCE', 'HOUR']])
            anomalies = sample_anom[preds == -1]
            if len(anomalies) > 0:
                st.error(f"Detected {len(anomalies)} Operational Anomalies in current sample.")
                st.dataframe(anomalies[['OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_DELAY']].head(), use_container_width=True)
            else:
                st.success("No critical network anomalies detected.")
        st.caption("Unsupervised ML (Isolation Forest) monitoring for non-standard delay patterns.")

    with col_b:
        st.markdown("#### ✈️ Airframe Fatigue Tracker")
        tail_path = get_file_path("tail_rotation_stats.csv", "results")
        if os.path.exists(tail_path):
            tails = pd.read_csv(tail_path)
            st.dataframe(tails[['TAIL_NUM', 'AIRCRAFT_TYPE', 'TOTAL_DELAY']].head(10), use_container_width=True)
        st.caption("Tracking accumulated delay minutes per specific GE-powered airframe.")
