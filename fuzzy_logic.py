import numpy as np
import skfuzzy as fuzz

# --- Tanımlar ---
x_wind = np.arange(0, 81, 1)
x_temp = np.arange(-40, 61, 1)
x_age  = np.arange(13, 101, 1)
x_humidity = np.arange(0, 101, 1)
x_rain = np.arange(0, 151, 1)
x_health = np.arange(0, 5.01, 0.01)

# --- Rüzgar üyelik fonksiyonları ---
# Kaynak: Beaufort skalası
def wind_mfs():
    return {
        'Sakin': fuzz.trimf(x_wind, [0, 0, 2]),
        'Hafif Esinti': fuzz.trimf(x_wind, [1, 3, 6]),
        'Hafif Rüzgar': fuzz.trimf(x_wind, [4, 7.5, 10]),
        'Zayıf Rüzgar': fuzz.trimf(x_wind, [8, 13, 17]),
        'Orta Kuvvetli Rüzgar': fuzz.trimf(x_wind, [15, 21.5, 27]),
        'Sert Rüzgar': fuzz.trimf(x_wind, [25, 31, 36]),
        'Kuvvetli Rüzgar': fuzz.trimf(x_wind, [34, 41.5, 47]),
        'Fırtına': fuzz.trimf(x_wind, [45, 53, 60]),
        'Kuvvetli Fırtına': fuzz.trimf(x_wind, [58, 66, 73]),
        'Tam Fırtına': fuzz.trimf(x_wind, [71, 79, 86]),
        'Şiddetli Fırtına': fuzz.trimf(x_wind, [84, 95, 105]),
        'Kasırga': fuzz.trapmf(x_wind, [102, 110, 150, 150])
    }



# --- Sıcaklık üyelik fonksiyonları ---
def temp_mfs():
    return {
        'Çok Soğuk': fuzz.trimf(x_temp, [-40, -30, -15]),
        'Soğuk': fuzz.trimf(x_temp, [-20, -5, 10]),
        'Ilıman': fuzz.trimf(x_temp, [5, 17.5, 30]),
        'Sıcak': fuzz.trimf(x_temp, [25, 35, 45]),
        'Aşırı Sıcak': fuzz.trapmf(x_temp, [42, 50, 60, 60])
    }

# --- Yaş üyelik fonksiyonları ---
def age_mfs():
    return {
        'Ergen': fuzz.trimf(x_age, [10, 15, 20]),
        'Genç Yetişkin': fuzz.trimf(x_age, [17, 26, 36]),
        'Orta Yaşlı': fuzz.trimf(x_age, [33, 45, 58]),
        'Yaşlı': fuzz.trimf(x_age, [55, 67, 80]),
        'Çok Yaşlı': fuzz.trapmf(x_age, [75, 85, 100, 100])
    }

    
# --- Nem üyelik fonksiyonları ---    
#def humidity_mfs():
#    return {
#        'Çok Kuru': fuzz.trimf(x_humidity, [0, 0, 20]),
#        'Kuru': fuzz.trimf(x_humidity, [15, 30, 45]),
#        'Konforlu': fuzz.trimf(x_humidity, [40, 50, 60]),
#        'Nemli': fuzz.trimf(x_humidity, [55, 70, 85]),
#        'Çok Nemli': fuzz.trapmf(x_humidity, [80, 90, 100, 100])
#    }
 
def humidity_mfs():
    return {
        'Uçta (Tehlikeli)': np.fmax(
            fuzz.trapmf(x_humidity, [0, 0, 10, 30]),
            fuzz.trapmf(x_humidity, [80, 90, 100, 100])
        ),
        'Orta (Konforlu)': fuzz.trimf(x_humidity, [30, 55, 80])
    } 
# --- Yağış üyelik fonksiyonları ---    
def rainfall_intensity_mfs():
    return {
        'Hiç Yağış Yok': fuzz.trimf(x_rain, [0, 0, 0.3]),
        'Hafif': fuzz.trimf(x_rain, [0.1, 1.5, 3]),
        'Orta': fuzz.trimf(x_rain, [2.5, 5.5, 9]),
        'Kuvvetli': fuzz.trimf(x_rain, [8, 22, 40]),
        'Şiddetli': fuzz.trimf(x_rain, [35, 65, 90]),
        'Aşırı': fuzz.trapmf(x_rain, [85, 100, 150, 150])
    }

    
# --- Sağlık üyelik fonksiyonları ---    

def health_status_mfs():
    return {
        'Sağlıklı': fuzz.trapmf(x_health, [0, 0, 0.5, 1.2]),
        'Orta': fuzz.trimf(x_health, [1.0, 1.7, 2.4]),
        'Riskli': fuzz.trimf(x_health, [2.1, 2.7, 3.3]),
        'Çok Riskli': fuzz.trimf(x_health, [3.0, 3.6, 4.2]),
        'Kritik': fuzz.trapmf(x_health, [4.0, 4.5, 5, 5])
    }


# --- Risk üyelik fonksiyonları (Çıkış için) ---
def risk_mfs():
    x_risk = np.arange(0, 1.01, 0.01)
    return {
        'Çok Düşük': fuzz.trimf(x_risk, [0.0, 0.1, 0.25]),
        'Düşük':     fuzz.trimf(x_risk, [0.15, 0.3, 0.45]),
        'Orta':      fuzz.trimf(x_risk, [0.35, 0.5, 0.65]),
        'Yüksek':    fuzz.trimf(x_risk, [0.55, 0.7, 0.85]),
        'Çok Yüksek':fuzz.trimf(x_risk, [0.75, 0.9, 1.0])
    }, x_risk


# --- Fuzzify + Defuzzify (bir değişken için) ---
def fuzzify_and_defuzzify(value, x_range, mfs):
    memberships = {label: fuzz.interp_membership(x_range, mf, value) for label, mf in mfs.items()}
    num = 0
    den = 0
    for label, degree in memberships.items():
        peak_x = x_range[np.argmax(mfs[label])]
        num += degree * peak_x
        den += degree
    return num / den if den != 0 else 0

# --- Normalizasyon fonksiyonu ---
def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

# --- Risk skor hesaplama fonksiyonu (Fuzzy çıkışlı) ---
def compute_risk_score_fuzzy(wind_val, temp_val, age_val, hum_val, rain_val, health_val, seatbelt_on):
    wind_norm = normalize(fuzzify_and_defuzzify(wind_val, x_wind, wind_mfs()), 0, 80)
    temp_norm = normalize(fuzzify_and_defuzzify(temp_val, x_temp, temp_mfs()), -40, 60)
    age_norm = normalize(fuzzify_and_defuzzify(age_val, x_age, age_mfs()), 13, 100)
    humidity_norm = normalize(fuzzify_and_defuzzify(hum_val, x_humidity, humidity_mfs()), 0, 100)
    rain_norm = normalize(fuzzify_and_defuzzify(rain_val, x_rain, rainfall_intensity_mfs()), 0, 150)
    health_norm = normalize(fuzzify_and_defuzzify(health_val, x_health, health_status_mfs()), 0, 5)
    
    vec = np.array([wind_norm, temp_norm, age_norm, humidity_norm, rain_norm, health_norm])
    norm_val = np.linalg.norm(vec)

    max_norm = np.linalg.norm([1, 1, 1, 1, 1, 1])

    # Önce normal risk skorunu hesapla (0-1 arası)
    risk_score = norm_val / max_norm

    # Eğer kemer takılı değilse %40 artır (maksimum 1.0 geçmesin)
    if not seatbelt_on:
        risk_score = min(risk_score * 1.45, 1.0)

    # Risk üyelik fonksiyonları ve x ekseni
    risk_funcs, x_risk = risk_mfs()

    # Risk skorunun her risk kümesindeki üyelik derecelerini hesapla
    memberships = {label: fuzz.interp_membership(x_risk, mf, risk_score) for label, mf in risk_funcs.items()}

    # Defuzzify işlemi (sayısal risk skoru)
    num = 0
    den = 0
    for label, degree in memberships.items():
        peak_x = x_risk[np.argmax(risk_funcs[label])]
        num += degree * peak_x
        den += degree
    risk_defuzzified = num / den if den != 0 else 0

    # Dilsel risk seviyesi: en yüksek üyelik dereceli küme
    risk_level = max(memberships, key=memberships.get)

    return {
        "score": round(risk_defuzzified, 3),        # Defuzzification sonucu
        "label": risk_level,                         # En yüksek üyelik dereceli fuzzy kategori
        "memberships": {k: round(v, 3) for k,v in memberships.items()},
        "defuzzified_score": round(risk_defuzzified, 3)
    }

# --- Örnek kullanım ---
if __name__ == "__main__":
    result = compute_risk_score_fuzzy(wind_val=80, temp_val=60, age_val=100, hum_val=100, rain_val=150, health_val=5, seatbelt_on=False)
    print("Risk Skoru (defuzzify sonucu 0-1 arası):", result["score"])
    print("Risk Seviyesi (fuzzy dilsel çıktı):", result["label"])
    print("Risk Üyelik Dereceleri:", result["memberships"])
    print("Defuzzify Edilmiş Risk Skoru:", result["defuzzified_score"])
