import requests
import argparse

def new_game(url):
  players, stacks = [], []
  while name := input(f"Player {len(players)}: "):
    players.append(name)
  while stack := input(f"Player {len(stacks)} chips: "):
    stacks.append(int(stack))
  assert len(players) == len(stacks), "Player amount mismatch."
  return requests.post(url + "new_game", json={"players": players, "stacks": stacks})

def update_state(url):
  action = float(input("Betsize: "))
  pos = int(input("Position: "))
  assert pos >= 0, "Invalid position."
  return requests.post(url + "update_state", json={"action": action, "pos": pos})

def update_board(url):
  board = input("Board: ")
  assert len(board) % 2 == 0 and len(board) >= 6, "Invalid board."
  return requests.post(url + "update_board", json={"board": board})

def solution(url):
  hand = input("Hand: ")
  assert len(hand) == 4, "Invalid hand."
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
  while inp := input("Endpoint: "):
    if matches := [e[1] for e in endpoints if e[0] == inp]:
      print(matches[0](root_url))
    else:
      print("Invalid endpoint.")
