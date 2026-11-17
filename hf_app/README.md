---
title: GE Aviation Delay Intel
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Aviation Delay Intelligence Platform ✈️

A production-grade decision support system for predicting and analyzing flight delays, targeting GE Aviation Digital Technology. This platform combines traditional Machine Learning with Generative AI to provide actionable operational insights.

## 🚀 Live Demo
[Link to be added after deployment]

## 🛠️ Features
- **Predictive Modeling**: XGBoost-based delay probability forecasting.
- **GenAI Insight Engine**: Converts complex ML scores into human-readable cockpit/dispatch briefs using Google Gemini.
- **Airport Vulnerability Index (AVI)**: Dynamic scoring of airports based on delay propagation risks.
- **Operational Dashboard**: Interactive Streamlit interface for fleet overview and real-time prediction.

## 📁 Project Structure
```
flight-delay-analysis/
├── data/              # BTS On-Time Performance data (sampled for demo)
├── src/               # Core analytics and modeling logic
│   ├── data_loader.py    # Data generation and ingestion
│   ├── preprocessing.py  # Feature engineering pipeline
│   ├── model_training.py # ML training (Logistic + XGBoost)
│   └── genai_engine.py   # Gemini AI integration
├── dashboard/         # Streamlit application
├── results/           # Saved models and performance metrics
└── requirements.txt
```

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd flight-delay-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file or export your API key:
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```

4. **Generate Data & Train Model** (if running locally for the first time):
   ```bash
   python src/data_loader.py
   python src/model_training.py
   ```

5. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Data Strategy
This project uses **Bureau of Transportation Statistics (BTS) On-Time Performance data**. For demonstration purposes, a high-fidelity synthetic sample (100k rows) following the BTS schema is provided. 

**Note on Cascade Analysis**: Since tail numbers are restricted in some public subsets, delay propagation is modeled using temporal and route proximity proxies (Same carrier/origin/hour).

## 🤖 AI Insight Engine
The platform uses the **Gemini 1.5 Flash** model to analyze model risk factors (Feature Importance) and flight details, generating operational briefs such as:
> "Flight DL102 has a 67% delay risk primarily driven by evening peak congestion at ATL and ULCC turnaround constraints. Recommend monitoring connection windows for downstream flights."

---
*Developed for GE Aviation Digital Technology recruitment - Demonstrating ML, Data Engineering, and GenAI capabilities.*
