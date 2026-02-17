import streamlit as st
from fuzzy_logic import compute_risk_score_fuzzy

st.set_page_config(page_title="Working at Height Risk Assessment System",
                   layout="centered")

st.title("Working at Height Risk Assessment System")
st.caption("Fuzzy Logic-Based Occupational Risk Decision Support Model")

# ---- SCENARIOS ----
scenarios = [
    {"name": "Scenario 1 – Young worker, mild outdoor conditions",
     "wind":10, "temp":15, "age":25, "humidity":60, "rain":0, "health":0.5, "belt":True},

    {"name": "Scenario 2 – Experienced worker, hot day",
     "wind":76, "temp":25, "age":45, "humidity":25, "rain":0, "health":1.0, "belt":True},

    {"name": "Scenario 3 – Elderly worker, humid weather",
     "wind":30, "temp":20, "age":60, "humidity":85, "rain":10, "health":1.5, "belt":True},

    {"name": "Scenario 4 – Storm & heavy rain, no harness",
     "wind":70, "temp":12, "age":40, "humidity":70, "rain":50, "health":1.0, "belt":False},

    {"name": "Scenario 5 – Elderly unhealthy worker, extreme heat",
     "wind":5, "temp":42, "age":68, "humidity":50, "rain":0, "health":4.0, "belt":True},
]

# ---- Scenario Selection ----
selected_scenario = st.selectbox(
    "Select a predefined risk scenario (optional):",
    ["Custom Input"] + [s["name"] for s in scenarios]
)

if selected_scenario != "Custom Input":
    scenario_data = next(s for s in scenarios if s["name"] == selected_scenario)
else:
    scenario_data = {"wind":10, "temp":20, "age":30, "humidity":60,
                     "rain":25, "health":3.0, "belt":False}

st.markdown("---")
st.subheader("Input Parameters")

wind = st.slider("Wind Speed (km/h)", 0, 80, scenario_data["wind"])
temp = st.slider("Temperature (°C)", -40, 60, scenario_data["temp"])
age = st.slider("Worker Age", 13, 100, scenario_data["age"])
humidity = st.slider("Relative Humidity (%)", 0, 100, scenario_data["humidity"])
rain = st.slider("Rainfall (mm)", 0, 150, scenario_data["rain"])
health = st.slider("Health Condition (0 = Healthy, 5 = Critical)", 0.0, 5.0, scenario_data["health"])
belt = st.toggle("Safety Harness Equipped", value=scenario_data["belt"])

st.markdown("---")

if st.button("Calculate Risk"):

    result = compute_risk_score_fuzzy(
        wind, temp, age, humidity, rain, health, belt
    )

    st.subheader("Risk Assessment Result")

    st.metric("Defuzzified Risk Score (0–1 Scale)", result["score"])
    st.write("**Linguistic Risk Level:**", result["label"])

    st.markdown("### Membership Degrees")
    st.json(result["memberships"])

    st.progress(result["score"])
