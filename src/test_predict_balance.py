import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "predict": "Balance Total",
    "Exports Total": 50000,
    "Imports Total": 56000,
    "Amount Amount": 3000,
    "Amount Inflation Rate": 2.0,
    "Actual Balance Total": -4700
}
response = requests.post(url, json=payload)
print("Response:", response.json())
