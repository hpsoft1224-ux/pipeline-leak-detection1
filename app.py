from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("final_hardware_pipeline_random_forest.pkl")

with open("final_deployment_config.json", "r") as f:
    config = json.load(f)

THRESHOLD = config["threshold_lpm"]


@app.route("/")
def home():
    return jsonify({
        "status": "Pipeline leak detection API running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    inlet = float(data["inlet_lpm"])
    outlet = float(data["outlet_lpm"])

    deficit = inlet - outlet

    features = np.array([
        [inlet, outlet, deficit]
    ])

    rf_prediction = int(model.predict(features)[0])

    rf_probability = float(
        model.predict_proba(features)[0][1]
    )

    threshold_alarm = int(deficit >= THRESHOLD)

    final_prediction=int(rf_prediction == 1 or threshold-alarm ==1)
    status = "LEAK" if final_prediction else "NO LEAK"

    return jsonify({
        "inlet_lpm": inlet,
        "outlet_lpm": outlet,
        "flow_deficit_lpm": deficit,
        "rf_prediction": rf_prediction,
        "rf_probability": rf_probability,
        "threshold_alarm": threshold_alarm,
        "final_prediction": final_prediction,
        "status": status
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
