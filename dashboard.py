import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="EIT Dashboard", layout="wide")
st.title("EIT Preventive Intelligence Dashboard")

col1, col2 = st.columns(2)

with col1:
    hours_awake = st.slider("Hours Awake", 0.0, 24.0, 16.0, 0.5)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 5.0, 0.5)
    eye_strain = st.slider("Eye Strain", 0.0, 1.0, 0.4, 0.05)
    reaction_time = st.slider("Reaction Time", 200.0, 1000.0, 420.0, 10.0)
    hydration = st.slider("Hydration", 0.0, 1.0, 0.6, 0.05)

with col2:
    heart_rate = st.slider("Heart Rate", 50.0, 140.0, 85.0, 1.0)
    steering_variability = st.slider("Steering Variability", 0.0, 1.0, 0.25, 0.05)
    lane_drift = st.slider("Lane Drift", 0.0, 1.0, 0.15, 0.05)
    time_of_day = st.slider("Time Of Day", 0.0, 24.0, 2.0, 0.5)
    food_intake = st.slider("Hours Since Last Meal", 0.0, 12.0, 4.0, 0.5)

payload = {
    "session_id": "dashboard-user",
    "hours_awake": hours_awake,
    "sleep_hours": sleep_hours,
    "eye_strain": eye_strain,
    "reaction_time": reaction_time,
    "hydration": hydration,
    "heart_rate": heart_rate,
    "steering_variability": steering_variability,
    "lane_drift": lane_drift,
    "time_of_day": time_of_day,
    "food_intake": food_intake,
}

if st.button("Predict Risk", use_container_width=True):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        pred = data.get("prediction", {})
        trend = data.get("trend", {})
        alerts = data.get("alerts", {})
        agent = data.get("agent", {})

        a, b, c, d = st.columns(4)
        a.metric("Risk", pred.get("Risk", "-"))
        b.metric("Status", pred.get("Status", "-"))
        c.metric("Trend", trend.get("Trend", "-"))
        d.metric("Dominant Factor", pred.get("DominantFactor", "-"))

        st.subheader("Recommended Action")
        action = pred.get("RecommendedAction", "No action available")
        if pred.get("Status") == "GREEN":
            st.success(action)
        elif pred.get("Status") == "YELLOW":
            st.warning(action)
        else:
            st.error(action)

        st.subheader("Alerts")
        for msg in alerts.get("messages", []):
            st.warning(msg)

        st.subheader("AI Agent")
        st.json(agent)

        st.subheader("Full Response")
        st.json(data)

    except Exception as e:
        st.error(f"API error: {e}")
