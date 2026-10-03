"""
app.py — Flask backend for Crop Yield Prediction
-------------------------------------------------
Dataset: Agriculture Precision Study Cases — Kaggle
Model:   Best-performing model selected from experimental results

Run:
    cd app/
    python3 app.py
Then open http://127.0.0.1:5000
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load saved model artifact ────────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
artifact      = joblib.load(ARTIFACT_PATH)
pipeline      = artifact['pipeline']
MODEL_NAME    = artifact['model_name']
MODEL_RMSE    = float(artifact['rmse'])
MODEL_R2      = float(artifact['r2'])
NUM_FEATURES  = artifact['num_features']
CAT_FEATURES  = artifact['cat_features']

# ── Valid categorical values (must match training data) ──────────────────────
VALID_CROPS      = ['Cotton', 'Dates', 'Maize', 'Mango', 'Potato',
                    'Rice', 'Soybean', 'Sugarcane', 'Tomato', 'Wheat']
VALID_SOILS      = ['Clay', 'Loam', 'Peaty', 'Saline', 'Sandy', 'Silty']
VALID_IRRIGATION = ['Drip', 'Flood Irrigation', 'Hydroponic',
                    'Rainfed', 'Smart Drip', 'Sprinkler']
VALID_REGIONS    = ['Africa', 'East Asia', 'Europe', 'Middle East',
                    'North America', 'Oceania', 'South America',
                    'South Asia', 'Southeast Asia']
VALID_FARM_TYPES = ['Greenhouse', 'Precision Farm', 'Semi-Mechanized',
                    'Smart AI Farm', 'Traditional', 'Vertical Farm']

print(f"Loaded model  : {MODEL_NAME}")
print(f"Model R²      : {MODEL_R2}")
print(f"Model RMSE    : {MODEL_RMSE} tons/ha")
print(f"Num features  : {NUM_FEATURES}")
print(f"Cat features  : {CAT_FEATURES}")


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        model_r2=MODEL_R2,
        model_rmse=MODEL_RMSE,
        crops=VALID_CROPS,
        soils=VALID_SOILS,
        irrigations=VALID_IRRIGATION,
        regions=VALID_REGIONS,
        farm_types=VALID_FARM_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # ── Read & validate inputs ─────────────────────────────────────────
        avg_temperature_c      = float(data.get("avg_temperature_c", 27))
        annual_rainfall_mm     = float(data.get("annual_rainfall_mm", 650))
        fertilizer_kg_per_ha   = float(data.get("fertilizer_kg_per_ha", 135))
        pesticide_litre_per_ha = float(data.get("pesticide_litre_per_ha", 2.4))
        soil_moisture_percent  = float(data.get("soil_moisture_percent", 37))
        climate_stress_index   = float(data.get("climate_stress_index", 47))

        crop            = str(data.get("crop", "Wheat"))
        soil_type       = str(data.get("soil_type", "Loam"))
        irrigation_type = str(data.get("irrigation_type", "Drip"))
        region          = str(data.get("region", "South Asia"))
        farm_type       = str(data.get("farm_type", "Traditional"))

        # ── Assemble DataFrame matching training feature order ──────────────
        row = {
            'avg_temperature_c':      avg_temperature_c,
            'annual_rainfall_mm':     annual_rainfall_mm,
            'fertilizer_kg_per_ha':   fertilizer_kg_per_ha,
            'pesticide_litre_per_ha': pesticide_litre_per_ha,
            'soil_moisture_percent':  soil_moisture_percent,
            'climate_stress_index':   climate_stress_index,
            'crop':                   crop,
            'soil_type':              soil_type,
            'irrigation_type':        irrigation_type,
            'region':                 region,
            'farm_type':              farm_type,
        }
        X = pd.DataFrame([row], columns=NUM_FEATURES + CAT_FEATURES)

        prediction = float(pipeline.predict(X)[0])
        prediction = round(max(0.0, prediction), 3)

        low  = round(max(0.0, prediction - MODEL_RMSE), 3)
        high = round(prediction + MODEL_RMSE, 3)

        return jsonify({
            "success":    True,
            "yield":      prediction,
            "error_low":  low,
            "error_high": high,
            "model_name": MODEL_NAME,
            "model_r2":   MODEL_R2,
            "model_rmse": MODEL_RMSE,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print(f"Starting server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
