import requests
import argparse

def cast(num_str, dt):
  try:
    return dt(num_str)
  except ValueError:
    print("Input is not a number.")
    return None

def new_game(url):
  stacks = []
  while stack := input(f"Player {len(stacks)} chips: "):
    if (v := cast(stack, int)) is not None:
      stacks.append(v)
  print(f"{stacks=}")
  return requests.post(url + "new_game", json={"stacks": stacks})

def update_state(url):
  if (action := cast(input("Betsize: "), float)) is None: return None
  if action < 0:
    print("Invalid betsize.")
    return None
  if (pos := cast(input("Position: "), int)) is None: return None
  if pos < 0:
    print("Invalid position")
    return None
  return requests.post(url + "update_state", json={"action": action, "pos": pos})

def update_board(url):
  board = input("Board: ")
  if len(board) % 2 != 0 or len(board) < 6:
    print("Invalid board.")
    return None
  return requests.post(url + "update_board", json={"board": board})

def solution(url):
  hand = input("Hand: ")
  if len(hand) != 4:
    print("Invalid hand.")
    return None
  return requests.post(url + "solution", json={"hand": hand})

endpoints = [
  ("new_game", new_game),
  ("update_state", update_state),
  ("update_board", update_board),
  ("solution", solution)
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("server", type=str)
  args = parser.parse_args()
  root_url = f"http://{args.server}:8080/"
  while inp := input("\nEndpoint: "):
    if matches := [e[1] for e in endpoints if e[0] == inp]:
      print(f"Response: {res.json() if (res := matches[0](root_url)) else res}")
    else:
      print("Invalid endpoint.")
