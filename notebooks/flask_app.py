import numpy as np
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names used during training
feature_names = ["Balance Total", "Exports Total", "Imports Total", "Amount", "Inflation Rate"]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.get_json()

    # Ensure the data matches the expected feature order
    try:
        # Create an array with the correct feature order and make predictions
        input_data = np.array([[
            data["Balance Total"][0],  # Ensure Balance Total is included
            data["Exports Total"][0],
            data["Imports Total"][0],
            data["Amount"][0],
            data["Inflation Rate"][0]
        ]])

        # Scale the data before prediction (if scaling was done during training)
        input_data_scaled = scaler.transform(input_data)

        # Get the prediction
        prediction = model.predict(input_data_scaled)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except KeyError as e:
        return jsonify({'error': f'Missing feature in request: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
