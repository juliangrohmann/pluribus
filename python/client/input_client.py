import requests
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("server", type=str)
  args = parser.parse_args()

  url = f"http://{args.server}:8080/new_game"
  payload = {
    "players": ["Alice", "Bob"],
    "stacks": [1000, 1000]
  }

  response = requests.post(url, json=payload)
  print(response.json())
