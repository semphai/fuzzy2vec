import numpy as np
import skfuzzy as fuzz

# ---------- Universe Definitions ----------

x_wind = np.arange(0, 81, 1)
x_temp = np.arange(-40, 61, 1)
x_age = np.arange(13, 101, 1)
x_humidity = np.arange(0, 101, 1)
x_rain = np.arange(0, 151, 1)
x_health = np.arange(0, 5.01, 0.01)

# ---------- Wind Membership Functions ----------

def wind_mfs():
    return {
        "Calm": fuzz.trimf(x_wind, [0, 0, 2]),
        "Light Breeze": fuzz.trimf(x_wind, [1, 3, 6]),
        "Light Wind": fuzz.trimf(x_wind, [4, 7.5, 10]),
        "Moderate Wind": fuzz.trimf(x_wind, [8, 13, 17]),
        "Strong Breeze": fuzz.trimf(x_wind, [15, 21.5, 27]),
        "Strong Wind": fuzz.trimf(x_wind, [25, 31, 36]),
        "Very Strong Wind": fuzz.trimf(x_wind, [34, 41.5, 47]),
        "Storm": fuzz.trimf(x_wind, [45, 53, 60]),
        "Severe Storm": fuzz.trimf(x_wind, [58, 66, 73]),
        "Full Storm": fuzz.trimf(x_wind, [71, 79, 86]),
        "Violent Storm": fuzz.trimf(x_wind, [84, 95, 105]),
        "Hurricane": fuzz.trapmf(x_wind, [102, 110, 150, 150])
    }

# ---------- Temperature ----------

def temp_mfs():
    return {
        "Very Cold": fuzz.trimf(x_temp, [-40, -30, -15]),
        "Cold": fuzz.trimf(x_temp, [-20, -5, 10]),
        "Mild": fuzz.trimf(x_temp, [5, 17.5, 30]),
        "Hot": fuzz.trimf(x_temp, [25, 35, 45]),
        "Extreme Heat": fuzz.trapmf(x_temp, [42, 50, 60, 60])
    }

# ---------- Age ----------

def age_mfs():
    return {
        "Adolescent": fuzz.trimf(x_age, [10, 15, 20]),
        "Young Adult": fuzz.trimf(x_age, [17, 26, 36]),
        "Middle Aged": fuzz.trimf(x_age, [33, 45, 58]),
        "Elderly": fuzz.trimf(x_age, [55, 67, 80]),
        "Very Elderly": fuzz.trapmf(x_age, [75, 85, 100, 100])
    }

# ---------- Humidity ----------

def humidity_mfs():
    return {
        "Extreme": np.fmax(
            fuzz.trapmf(x_humidity, [0, 0, 10, 30]),
            fuzz.trapmf(x_humidity, [80, 90, 100, 100])
        ),
        "Comfortable": fuzz.trimf(x_humidity, [30, 55, 80])
    }

# ---------- Rain ----------

def rainfall_mfs():
    return {
        "No Rain": fuzz.trimf(x_rain, [0, 0, 0.3]),
        "Light": fuzz.trimf(x_rain, [0.1, 1.5, 3]),
        "Moderate": fuzz.trimf(x_rain, [2.5, 5.5, 9]),
        "Heavy": fuzz.trimf(x_rain, [8, 22, 40]),
        "Severe": fuzz.trimf(x_rain, [35, 65, 90]),
        "Extreme": fuzz.trapmf(x_rain, [85, 100, 150, 150])
    }

# ---------- Health ----------

def health_mfs():
    return {
        "Healthy": fuzz.trapmf(x_health, [0, 0, 0.5, 1.2]),
        "Moderate": fuzz.trimf(x_health, [1.0, 1.7, 2.4]),
        "Risky": fuzz.trimf(x_health, [2.1, 2.7, 3.3]),
        "Very Risky": fuzz.trimf(x_health, [3.0, 3.6, 4.2]),
        "Critical": fuzz.trapmf(x_health, [4.0, 4.5, 5, 5])
    }

# ---------- Risk Output ----------

def risk_mfs():
    x_risk = np.arange(0, 1.01, 0.01)
    return {
        "Very Low": fuzz.trimf(x_risk, [0.0, 0.1, 0.25]),
        "Low": fuzz.trimf(x_risk, [0.15, 0.3, 0.45]),
        "Medium": fuzz.trimf(x_risk, [0.35, 0.5, 0.65]),
        "High": fuzz.trimf(x_risk, [0.55, 0.7, 0.85]),
        "Very High": fuzz.trimf(x_risk, [0.75, 0.9, 1.0])
    }, x_risk

# ---------- Core Functions ----------

def fuzzify_and_defuzzify(value, x_range, mfs):
    memberships = {label: fuzz.interp_membership(x_range, mf, value) for label, mf in mfs.items()}
    num = 0
    den = 0
    for label, degree in memberships.items():
        peak_x = x_range[np.argmax(mfs[label])]
        num += degree * peak_x
        den += degree
    return num / den if den != 0 else 0

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def compute_risk_score_fuzzy(wind_val, temp_val, age_val, hum_val, rain_val, health_val, seatbelt_on):

    wind_norm = normalize(fuzzify_and_defuzzify(wind_val, x_wind, wind_mfs()), 0, 80)
    temp_norm = normalize(fuzzify_and_defuzzify(temp_val, x_temp, temp_mfs()), -40, 60)
    age_norm = normalize(fuzzify_and_defuzzify(age_val, x_age, age_mfs()), 13, 100)
    humidity_norm = normalize(fuzzify_and_defuzzify(hum_val, x_humidity, humidity_mfs()), 0, 100)
    rain_norm = normalize(fuzzify_and_defuzzify(rain_val, x_rain, rainfall_mfs()), 0, 150)
    health_norm = normalize(fuzzify_and_defuzzify(health_val, x_health, health_mfs()), 0, 5)

    vec = np.array([wind_norm, temp_norm, age_norm, humidity_norm, rain_norm, health_norm])
    norm_val = np.linalg.norm(vec)
    max_norm = np.linalg.norm([1, 1, 1, 1, 1, 1])

    risk_score = norm_val / max_norm

    if not seatbelt_on:
        risk_score = min(risk_score * 1.45, 1.0)

    risk_funcs, x_risk = risk_mfs()
    memberships = {label: fuzz.interp_membership(x_risk, mf, risk_score) for label, mf in risk_funcs.items()}

    num = 0
    den = 0
    for label, degree in memberships.items():
        peak_x = x_risk[np.argmax(risk_funcs[label])]
        num += degree * peak_x
        den += degree

    risk_defuzzified = num / den if den != 0 else 0
    risk_level = max(memberships, key=memberships.get)

    return {
        "score": round(risk_defuzzified, 3),
        "label": risk_level,
        "memberships": {k: round(v, 3) for k, v in memberships.items()}
    }
