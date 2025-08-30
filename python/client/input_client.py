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
  print(args)
  stacks = []
  if len(hero_hand := (input("Hand: ") if not args else args.pop(0)).strip()) != 4: return print("Invalid hand.")
  if (hero_pos := cast(input("Hero position: ") if args is None else args.pop(0), int)) is None: return None
  if hero_pos < 0: return print("Invalid position.")
  while stack := (input(f"Player {len(stacks)} chips: ") if args is None else args.pop(0) if args else None):
    if (v := cast(stack, int)) is not None: stacks.append(v)
  if hero_pos >= len(stacks): print("Invalid position.")
  return to_url(host, "new_game"), {"stacks": stacks, "hero_hand": hero_hand, "hero_pos": hero_pos}

def update_state(host, args=None):
  if (action := cast(input("Betsize: ") if not args else args.pop(0), float)) is None: return None
  if (pos := cast(input("Position: ") if not args else args.pop(0), int)) is None: return None
  if pos < 0: return print("Invalid position")
  return to_url(host, "update_state"), {"action": action, "pos": pos}

def hero_action(host, args=None):
  freq = []
  if (action := cast(input("Betsize: ") if not args else args.pop(0), float)) is None: return None
  while f_str := (input(f"Action {len(freq)} frquency: ") if args is None else args.pop(0) if args else None):
    if (v := cast(f_str, float)) is not None: freq.append(v)
  return to_url(host, "hero_action"), {"action": action, "freq": freq}

def update_board(host, args=None):
  board = (input("Board: ") if not args else args[0]).strip()
  return print("Invalid board.") if len(board) % 2 != 0 or len(board) < 6 else to_url(host, "update_board"), {"board": board}

def solution(host, args=None):
  hand = (input("Hand: ") if not args else args[0]).strip()
  return print("Invalid hand.") if len(hand) != 4 else to_url(host, "solution"), {"hand": hand}

def save_range(host, args=None):
  fn = (input("Filename: ") if not args else args[0]).strip()
  return to_url(host, "save_range"), {"fn": fn}

def view_range(host, args=None):
  fn = (input("Filename: ") if not args else args.pop(0)).strip()
  duration = cast(input("Duration (s): ") if not args else args.pop(0), float)
  t_0, viewing = time.time(), False
  try:
    while time.time() - t_0 < duration:
      save_range(host, [fn])
      subprocess.run(["scp", f"root@{host}:/root/pluribus/build/{fn}", fn])
      if not viewing: viewing, _ = True, subprocess.run(["xdg-open", fn])
  except KeyboardInterrupt:
    print(f"\nInterrupted range viewing: {time.time() - t_0:.2f} / {duration:.2f} s")

def wait(_, args=None):
  if args:
    t_0, duration = time.time(), cast(args[0], float)
    try:
      if duration: time.sleep(duration)
      else: print(f"Invalid duration: {args[0]}")
    except KeyboardInterrupt:
      print(f"\nInterrupted waiting: {time.time() - t_0:.2f} / {duration:.2f} seconds")

def dispatch_endpoint(host, args, endpoints):
  if matches := [e[1] for e in endpoints if e[0] == args[0]]:
    try: print(f"Response: {requests.post(payload[0], json=payload[1]).json() if (payload := matches[0](host, args[1:] if len(args) > 1 else None)) else payload}")
    except requests.exceptions.ConnectionError: print(f"Connection refused.")
  else: print("Invalid endpoint.")

endpoints = [
  ("new_game", new_game),
  ("update_state", update_state),
  ("hero_action", hero_action),
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
          print(line.strip())
          if not line.lstrip()[0] == '%': dispatch_endpoint(cmd_args.host, line.strip().split(' '), endpoints)
    else:
      dispatch_endpoint(cmd_args.host, inp_args, endpoints)


