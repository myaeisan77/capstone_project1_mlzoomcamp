import requests


url = "http://localhost:9697/predict"

client = {"Location": "Alabama", "Category": "Clothing" }


response = requests.post(url, json=client).json()
print(response)
 