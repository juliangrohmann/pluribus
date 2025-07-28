import requests

url = "http://5.39.216.171:8080/new_game"
payload = {
    "players": ["Alice", "Bob"],
    "stacks": [1000, 1000]
}

response = requests.post(url, json=payload)
print(response.json())
