import streamlit as st
from fuzzy_logic import compute_risk_score_fuzzy

st.set_page_config(page_title="Working at Height Risk Assessment System",
                   layout="centered")

st.title("Working at Height Risk Assessment System")
st.markdown("### Fuzzy-Based Decision Support Model")

st.markdown("---")

# Input Section
st.subheader("Input Parameters")

wind = st.slider("Wind Speed (km/h)", 0, 80, 10)
temp = st.slider("Temperature (°C)", -40, 60, 20)
age = st.slider("Worker Age", 13, 100, 30)
humidity = st.slider("Relative Humidity (%)", 0, 100, 60)
rain = st.slider("Rainfall (mm)", 0, 150, 25)
health = st.slider("Health Condition (0 = Healthy, 5 = Critical)", 0.0, 5.0, 3.0)
belt = st.toggle("Safety Harness Equipped")

st.markdown("---")

if st.button("Calculate Risk"):

    result = compute_risk_score_fuzzy(
        wind, temp, age, humidity, rain, health, belt
    )

    st.subheader("Risk Assessment Result")

    st.metric("Defuzzified Risk Score (0–1 Scale)", result["score"])
    st.write("**Linguistic Risk Level:**", result["label"])

    st.markdown("#### Membership Degrees")
    st.json(result["memberships"])

    # Risk visualization bar
    st.progress(result["score"])
