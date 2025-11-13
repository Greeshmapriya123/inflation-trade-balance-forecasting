import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "predict": "Amount Inflation Rate",
    "Exports Total": 50000,
    "Imports Total": 56000,
    "Amount Amount": 3000,
    "Balance Total": -4700,
    "Actual Amount Inflation Rate": 2.0
}

response = requests.post(url, json=payload)
print("Response from API:", response.json())



