import requests

res = requests.get("http://localhost:8000/api/incidents/C000014")
if res.status_code == 200:
    print(res.json().get('model_prediction'))
else:
    print(f"Error {res.status_code}")
