from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
LOG_FILE = "logs.csv"

# Define model mapping
model_files = {
    "Balance Total": ("model_balance.pkl", "imputer_balance.pkl", "scaler_balance.pkl", ["Exports Total", "Imports Total", "Amount Amount", "Amount Inflation Rate"]),
    "Exports Total": ("model_exports.pkl", "imputer_exports.pkl", "scaler_exports.pkl", ["Imports Total", "Amount Amount", "Amount Inflation Rate", "Balance Total"]),
    "Imports Total": ("model_imports.pkl", "imputer_imports.pkl", "scaler_imports.pkl", ["Exports Total", "Amount Amount", "Amount Inflation Rate", "Balance Total"]),
    "Amount Amount": ("model_amount.pkl", "imputer_amount.pkl", "scaler_amount.pkl", ["Exports Total", "Imports Total", "Amount Inflation Rate", "Balance Total"]),
    "Amount Inflation Rate": ("model_inflation.pkl", "imputer_inflation.pkl", "scaler_inflation.pkl", ["Exports Total", "Imports Total", "Amount Amount", "Balance Total"]),
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        target = data.get("predict")

        if target not in model_files:
            return jsonify({"error": f"Unsupported prediction target: {target}"}), 400

        model_file, imputer_file, scaler_file, input_fields = model_files[target]

        model = joblib.load(model_file)
        imputer = joblib.load(imputer_file)
        scaler = joblib.load(scaler_file)

        # Get input features
        X = np.array([data[field] for field in input_fields]).reshape(1, -1)
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        prediction = model.predict(X_scaled)[0]

        # Create log entry
        log_entry = {
            "Timestamp": datetime.now().isoformat(),
            "Prediction Target": target,
            "Predicted Value": round(prediction, 2)
        }

        for field in input_fields:
            log_entry[field] = data[field]

        actual_field = f"Actual {target}"
        if actual_field in data:
            log_entry[actual_field] = data[actual_field]

        df_log = pd.DataFrame([log_entry])
        if not os.path.exists(LOG_FILE):
            df_log.to_csv(LOG_FILE, index=False)
        else:
            df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)

        return jsonify({f"Predicted {target}": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
