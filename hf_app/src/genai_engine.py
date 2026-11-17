import google.generativeai as genai
import os
import json

def get_delay_insight(flight_details, prediction_prob, risk_factors):
    """
    Uses Gemini AI to generate a human-readable operational insight 
    based on model predictions and flight details.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    prompt = f"""
    You are an Aviation Operational Intelligence AI. 
    Analyze the following flight delay prediction and provide a concise operational brief for a cockpit or dispatch crew.
    
    Flight Details:
    - Carrier: {flight_details.get('carrier')}
    - Route: {flight_details.get('origin')} to {flight_details.get('dest')}
    - Departure Time: {flight_details.get('dep_time')}
    
    Model Prediction:
    - Probability of Delay (>15 mins): {prediction_prob:.1%}
    - Top Risk Factors: {', '.join(risk_factors)}
    
    Instructions:
    - Provide a professional, concise explanation (2-3 sentences).
    - Mention potential business implications (e.g., connection risk).
    - Format as a clear 'Operational Brief'.
    """

    if not api_key:
        # Mock Response for Demo Mode
        return f"DEMO MODE: {flight_details.get('carrier')} flight from {flight_details.get('origin')} to {flight_details.get('dest')} has a {prediction_prob:.1%} delay risk. Primary drivers: {', '.join(risk_factors)}. Recommend monitoring connection windows."

    try:
        genai.configure(api_key=api_key)
        # Using the exact name from diagnostics with models/ prefix
        model = genai.GenerativeModel('models/gemini-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback to gemini-pro-latest
        if "404" in str(e):
            try:
                model = genai.GenerativeModel('models/gemini-pro-latest')
                response = model.generate_content(prompt)
                return response.text
            except:
                pass
        return f"Insight Engine Error: {str(e)}"

if __name__ == "__main__":
    # Test
    sample_flight = {"carrier": "DL", "origin": "ATL", "dest": "ORD", "dep_time": "18:00"}
    factors = ["Late Aircraft Arrival", "Evening Peak Congestion"]
    print(get_delay_insight(sample_flight, 0.67, factors))
