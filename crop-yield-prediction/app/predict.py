"""
predict.py — Simple CLI prediction tool
Usage:
    cd app/
    python3 predict.py
"""

import joblib
import pandas as pd
import os

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

def main():
    artifact = joblib.load(ARTIFACT_PATH)
    pipeline = artifact['pipeline']
    print(f"\n=== Crop Yield Predictor (CLI) ===")
    print(f"Model: {artifact['model_name']}  R²={artifact['r2']}  RMSE={artifact['rmse']} t/ha\n")

    # Categorical options
    crops      = ['Cotton','Dates','Maize','Mango','Potato','Rice','Soybean','Sugarcane','Tomato','Wheat']
    soils      = ['Clay','Loam','Peaty','Saline','Sandy','Silty']
    irrigs     = ['Drip','Flood Irrigation','Hydroponic','Rainfed','Smart Drip','Sprinkler']
    regions    = ['Africa','East Asia','Europe','Middle East','North America',
                  'Oceania','South America','South Asia','Southeast Asia']
    farm_types = ['Greenhouse','Precision Farm','Semi-Mechanized','Smart AI Farm','Traditional','Vertical Farm']

    def pick(prompt, options, default):
        print(f"  Options: {options}")
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val in options else default

    print("Enter values (press Enter to use defaults):\n")
    temp     = float(input("  Average Temperature °C [27]: ") or 27)
    rain     = float(input("  Annual Rainfall mm [650]: ") or 650)
    fert     = float(input("  Fertilizer kg/ha [135]: ") or 135)
    pest     = float(input("  Pesticide litres/ha [2.4]: ") or 2.4)
    moisture = float(input("  Soil Moisture % [37]: ") or 37)
    stress   = float(input("  Climate Stress Index (0-100) [47]: ") or 47)
    crop     = pick("Crop", crops, "Wheat")
    soil     = pick("Soil Type", soils, "Loam")
    irr      = pick("Irrigation Type", irrigs, "Drip")
    region   = pick("Region", regions, "South Asia")
    farm     = pick("Farm Type", farm_types, "Traditional")

    row = {
        'avg_temperature_c':      temp,
        'annual_rainfall_mm':     rain,
        'fertilizer_kg_per_ha':   fert,
        'pesticide_litre_per_ha': pest,
        'soil_moisture_percent':  moisture,
        'climate_stress_index':   stress,
        'crop':                   crop,
        'soil_type':              soil,
        'irrigation_type':        irr,
        'region':                 region,
        'farm_type':              farm,
    }
    X = pd.DataFrame([row], columns=artifact['num_features'] + artifact['cat_features'])
    pred = pipeline.predict(X)[0]
    rmse = artifact['rmse']
    print(f"\n  Predicted Yield : {pred:.3f} tons/ha")
    print(f"  Expected Error  : ±{rmse} tons/ha  (one test-set RMSE)")
    print(f"  Likely Range    : {max(0, pred - rmse):.3f} – {pred + rmse:.3f} tons/ha\n")

if __name__ == '__main__':
    main()
