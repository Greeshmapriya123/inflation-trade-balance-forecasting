import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "predict": "Imports Total",
    "Exports Total": 50000,
    "Amount Amount": 3000,
    "Amount Inflation Rate": 2.0,
    "Balance Total": -4700,
    "Actual Imports Total": 56000
}

response = requests.post(url, json=payload)
print("Response from API:", response.json())

