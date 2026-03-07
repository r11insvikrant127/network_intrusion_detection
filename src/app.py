import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline
model = joblib.load("models/final_model.pkl")

st.title("Network Intrusion Detection System")

st.write("Enter Network Features")

# Example features (replace with actual UNSW feature names)
duration = st.number_input("Duration", value=0.0)
src_bytes = st.number_input("Source Bytes", value=0.0)
dst_bytes = st.number_input("Destination Bytes", value=0.0)

# For categorical
proto = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
service = st.selectbox("Service", ["http", "ftp", "dns"])
state = st.selectbox("State", ["FIN", "INT", "CON"])

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "duration": [duration],
        "src_bytes": [src_bytes],
        "dst_bytes": [dst_bytes],
        "proto": [proto],
        "service": [service],
        "state": [state]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Attack Detected")
    else:
        st.success("✅ Normal Traffic")