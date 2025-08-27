import requests
import argparse
import time

def cast(num_str, dt):
  try:
    return dt(num_str)
  except ValueError:
    print("Input is not a number.")
    return None

def new_game(url, args=None):
  stacks = []
  while stack := (input(f"Player {len(stacks)} chips: ") if args is None else args.pop(0) if args else None):
    if (v := cast(stack, int)) is not None:
      stacks.append(v)
  print(f"{stacks=}")
  return requests.post(url + "new_game", json={"stacks": stacks})

def update_state(url, args=None):
  if (action := cast(input("Betsize: ") if not args else args.pop(0), float)) is None: return None
  if action < 0:
    print("Invalid betsize.")
    return None
  if (pos := cast(input("Position: ") if not args else args.pop(0), int)) is None: return None
  if pos < 0:
    print("Invalid position")
    return None
  return requests.post(url + "update_state", json={"action": action, "pos": pos})

def update_board(url, args=None):
  board = (input("Board: ") if not args else args[0]).strip()
  if len(board) % 2 != 0 or len(board) < 6:
    print("Invalid board.")
    return None
  return requests.post(url + "update_board", json={"board": board})

def solution(url, args=None):
  hand = (input("Hand: ") if not args else args[0]).strip()
  if len(hand) != 4:
    print("Invalid hand.")
    return None
  return requests.post(url + "solution", json={"hand": hand})

def save_range(url, args=None):
  fn = (input("Filename: ") if not args else args[0]).strip()
  return requests.post(url + "save_range", json={"fn": fn})

def wait(_, args=None):
  if args: time.sleep(cast(args[0], float))

def dispatch_endpoint(args, endpoints):
  if matches := [e[1] for e in endpoints if e[0] == args[0]]:
    print(f"Response: {res.json() if (res := matches[0](root_url, args[1:])) else res}")
  else:
    print("Invalid endpoint.")

endpoints = [
  ("new_game", new_game),
  ("update_state", update_state),
  ("update_board", update_board),
  ("solution", solution),
  ("save_range", save_range),
  ("wait", wait),
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("server", type=str)
  cmd_args = parser.parse_args()
  root_url = f"http://{cmd_args.server}:8080/"
  while cmd_args := input("\nEndpoint: ").split(' '):
    if cmd_args and cmd_args[0] == "config":
      with open(cmd_args[1]) as f:
        for line in f: dispatch_endpoint(line.split(' '), endpoints)
    else:
      dispatch_endpoint(cmd_args, endpoints)


