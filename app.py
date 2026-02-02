from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/crop_yield_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

def prepare_input(rainfall, temperature, days, fertilizer, irrigation, crop, soil, weather, region):
    
    input_data = {col: 0 for col in feature_columns}
    input_data['Rainfall_mm'] = rainfall
    input_data['Temperature_Celsius'] = temperature
    input_data['Days_to_Harvest'] = days
    input_data['Fertilizer_Used'] = fertilizer
    input_data['Irrigation_Used'] = irrigation

    crop_col = f"Crop_{crop}"
    soil_col = f"Soil_Type_{soil}"
    weather_col = f"Weather_Condition_{weather}"
    region_col = f"Region_{region}"

    for col in [crop_col,soil_col,weather_col,region_col]:
        if col in input_data:
            input_data[col]=1

    X_input = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_input)

    return X_scaled


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        days = int(request.form["days"])
        fertilizer = int(request.form["fertilizer"])
        irrigation = int(request.form["irrigation"])
        crop = request.form["crop"]
        soil = request.form["soil"]
        weather = request.form["weather"]
        region = request.form["region"]

        X_ready = prepare_input(rainfall, temperature, days, fertilizer, irrigation, crop, soil, weather, region)
        prediction = round(model.predict(X_ready)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
