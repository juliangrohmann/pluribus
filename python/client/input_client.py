import requests
import subprocess
import argparse
import time

def to_url(host, endpoint): return f"http://{host}:8080/{endpoint}"

def cast(num_str, dt):
  try:
    return dt(num_str)
  except ValueError:
    return print("Input is not a number.")

def new_game(host, args=None):
  stacks = []
  while stack := (input(f"Player {len(stacks)} chips: ") if args is None else args.pop(0) if args else None):
    if (v := cast(stack, int)) is not None: stacks.append(v)
  return requests.post(to_url(host, "new_game"), json={"stacks": stacks})

def update_state(host, args=None):
  if (action := cast(input("Betsize: ") if not args else args.pop(0), float)) is None: return None
  if action < 0: return print("Invalid betsize.")
  if (pos := cast(input("Position: ") if not args else args.pop(0), int)) is None: return None
  if pos < 0: return print("Invalid position")
  return requests.post(to_url(host, "update_state"), json={"action": action, "pos": pos})

def update_board(host, args=None):
  board = (input("Board: ") if not args else args[0]).strip()
  return print("Invalid board.") if len(board) % 2 != 0 or len(board) < 6 else requests.post(to_url(host, "update_board"), json={"board": board})

def solution(host, args=None):
  hand = (input("Hand: ") if not args else args[0]).strip()
  return print("Invalid hand.") if len(hand) != 4 else requests.post(to_url(host, "solution"), json={"hand": hand})

def save_range(host, args=None):
  fn = (input("Filename: ") if not args else args[0]).strip()
  return requests.post(to_url(host, "save_range"), json={"fn": fn})

def view_range(host, args=None):
  fn = (input("Filename: ") if not args else args.pop(0)).strip()
  duration = cast(input("Duration (s): ") if not args else args.pop(0), float)
  t_0, viewing = time.time(), False
  while time.time() - t_0 < duration:
    save_range(host, [fn])
    subprocess.run(["scp", f"root@{host}:/root/pluribus/build/{fn}", fn])
    if not viewing: viewing, _ = True, subprocess.run(["xdg-open", fn])

def wait(_, args=None):
  if args: time.sleep(cast(args[0], float))

def dispatch_endpoint(host, args, endpoints):
  if matches := [e[1] for e in endpoints if e[0] == args[0]]:
    try: print(f"Response: {res.json() if (res := matches[0](host, args[1:] if len(args) > 1 else None)) else res}")
    except requests.exceptions.ConnectionError: print(f"Connection refused.")
  else: print("Invalid endpoint.")

endpoints = [
  ("new_game", new_game),
  ("update_state", update_state),
  ("update_board", update_board),
  ("solution", solution),
  ("save_range", save_range),
  ("view_range", view_range),
  ("wait", wait),
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("host", type=str)
  cmd_args = parser.parse_args()
  while inp_args := input("\nEndpoint: ").strip().split(' '):
    if inp_args and inp_args[0] == "config":
      with open(inp_args[1]) as f:
        for line in f:
          if not line.lstrip()[0] == '%': dispatch_endpoint(cmd_args.host, line.split(' '), endpoints)
    else:
      dispatch_endpoint(cmd_args.host, inp_args, endpoints)


