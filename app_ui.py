import streamlit as st
import pandas as pd
import requests

# 1. Dashboard Frame Window Configuration
st.set_page_config(page_title="India Property Valuation SaaS", page_icon="🏢", layout="centered")
st.title("🏢 Indian Residential Property Valuation Engine")
st.markdown("### Production-Grade MLOps SaaS Portal")
st.markdown("Enter property specifications below to fetch a real-time appraisal from our cloud-hosted FastAPI backend microservice.")

# 2. Form Input Layout Matrix
bhk = st.number_input("BHK Configuration", min_value=1, max_value=10, value=3)
sqft = st.number_input("Total Space Area (Square Feet)", min_value=150, max_value=100000, value=1500)
year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2020)
prop_type = st.selectbox("Property Type", ["Flat", "Independent House", "Villa", "Penthouse", "Apartment"])

# Using standard drop downs to prevent user spelling errors
state = st.selectbox("State Territory", ["Tamil Nadu", "Maharashtra", "Karnataka", "Delhi", "Gujarat", "West Bengal", "Telangana"])
city = st.selectbox("Metropolitan City", ["Chennai", "Mumbai", "Bangalore", "New Delhi", "Ahmedabad", "Kolkata", "Hyderabad", "Pune", "Gurgaon"])
locality = st.text_input("Locality Zone", value="Main Downtown Hub")

# 3. Your Official Live Render API Endpoint Link
BACKEND_API_URL = "https://indian-real-state-housing-valuation-97ol.onrender.com/api/v1/predict/"
if st.button("Calculate Market Valuation", type="primary"):
    # Group inputs into the strict JSON schema expected by your FastAPI backend
    payload_packet = {
        "BHK": int(bhk),
        "Size_in_SqFt": float(sqft),
        "Year_Built": int(year_built),
        "Property_Type": prop_type,
        "States": state,
        "City": city,
        "Locality": locality
    }
    
    with st.spinner("Streaming data packets to cloud compute clusters for predictive valuation..."):
        try:
            # Send the request over the internet to Render
            api_response = requests.post(BACKEND_API_URL, json=payload_packet)
            
            if api_response.status_code == 200:
                result_data = api_response.json()
                
                # Extract clean output variables from the JSON response
                formatted_price = result_data["valuation"]
                latency_speed = result_data["telemetry"]["latency_ms"]
                
                st.markdown("---")
                st.metric(label="Estimated Asset Market Value Quote", value=formatted_price)
                st.info(f"⚡ MLOps Telemetry Report: Live roundtrip computational latency scored at: {latency_speed} ms")
            else:
                st.error(f"⚠️ Gateway Error {api_response.status_code}: {api_response.text}")
                st.info("Check your FastAPI code configurations to ensure parameter mapping names match perfectly.")
        except Exception as network_failure:
            st.error(f"🔴 Connection Blocked: Unable to communicate with the FastAPI backend engine.")
            st.caption(f"Diagnostic Error Details: {network_failure}")
