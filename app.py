import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("AI Disease Predictor")
st.warning("DISCLAIMER: This is a machine learning project for educational purposes. Do not use for actual medical diagnosis.")
st.write("Select the symptoms you are experiencing to get a prediction.")

with st.form("prediction_form"):
    st.header("Symptom Checker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fever = st.radio("Do you have a fever?", ("No", "Yes"))
        headache = st.radio("Do you have a headache?", ("No", "Yes"))
        nausea = st.radio("Do you feel nauseous?", ("No", "Yes"))
        vomiting = st.radio("Are you vomiting?", ("No", "Yes"))
        fatigue = st.radio("Are you feel fatigue?", ("No", "Yes"))
        
    with col2:
        joint_pain = st.radio("Do you have joint pain?", ("No", "Yes"))
        skin_rash = st.radio("Do you have a skin rash?", ("No", "Yes"))
        cough = st.radio("Do you have a cough?", ("No", "Yes"))
        weight_loss = st.radio("Have you lost weight recently?", ("No", "Yes"))
        yellow_eyes = st.radio("Do you have yellow eyes?", ("No", "Yes"))
        
    def val(choice):
        return 1 if choice == "Yes" else 0
    
    submit_button = st.form_submit_button("Predict Disease")
    
    if submit_button:
        payload = {
            "fever": val(fever),
            "headache": val(headache),
            "nausea": val(nausea),
            "vomiting": val(vomiting),
            "fatigue": val(fatigue),
            "joint_pain": val(joint_pain),
            "skin_rash": val(skin_rash),
            "cough": val(cough),
            "weight_loss": val(weight_loss),
            "yellow_eyes": val(yellow_eyes)
        }
        
        with st.spinner("Analyzing symptoms..."):
            try:
                response = requests.post(API_URL, json = payload)
                
                if response.status_code == 200:
                    result = response.json()
                    disease = result['disease_name']
                    
                    confidence = result.get('confidence', 0) * 100
                    
                    st.success(f"### Predicted Disease: {disease} (Confidence: {confidence:.2f}%)")
                    
                    if confidence < 0.5:
                        st.warning(f"Confidence is low ({confidence:.1%}). The model is unsure.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is the FastAPI server running?")