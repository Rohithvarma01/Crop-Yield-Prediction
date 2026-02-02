import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/crop_yield_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

def prepare_input(rainfall, temperature, days, fertilizer, irrigation, crop, soil, weather, region):
    
    input_data = {col: 0 for col in feature_columns}
    input_data["Rainfall_mm"] = rainfall
    input_data["Temperature_Celsius"] = temperature
    input_data["Days_to_Harvest"] = days
    input_data["Fertilizer_Used"] = fertilizer
    input_data["Irrigation_Used"] = irrigation

    # One-hot encoded categorical features
    crop_col = f"Crop_{crop}"
    soil_col = f"Soil_Type_{soil}"
    weather_col = f"Weather_Condition_{weather}"
    region_col = f"Region_{region}"

    for col in [crop_col, soil_col, weather_col, region_col]:
        if col in input_data:
            input_data[col] = 1

    X_input = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_input)

    return X_scaled


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

st.title("🌾 Crop Yield Prediction")

st.write("Enter the following details to predict crop yield")

# Inputs (MATCHED WITH HTML FORM)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", step=0.1)
days = st.number_input("Days to Harvest", min_value=1, step=1)

fertilizer = st.selectbox("Fertilizer Used", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
irrigation = st.selectbox("Irrigation Used", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

region = st.selectbox("Region", ["North", "South", "West"])

soil = st.selectbox("Soil Type", ["Clay", "Loam", "Peaty", "Sandy", "Silt"])

crop = st.selectbox("Crop Type", ["Cotton", "Maize", "Rice", "Soybean", "Wheat"])

weather = st.selectbox("Weather Condition", ["Rainy", "Sunny"])

# Prediction
if st.button("Predict Yield"):
    X_ready = prepare_input(
        rainfall, temperature, days,
        fertilizer, irrigation,
        crop, soil, weather, region
    )

    prediction = round(model.predict(X_ready)[0], 2)
    st.success(f"🌱 Predicted Crop Yield: {prediction} tons/hectare")
