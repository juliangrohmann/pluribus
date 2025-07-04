import argparse
import pathlib
import os
import shutil
import json
import time
import wandb

def read_and_move(file_path, logged_dir):
  try:
    with open(file_path, 'r') as f:
      json_data = json.load(f)
    print(f"Processed {file_path}")
  except json.JSONDecodeError as e:
    print(f"Skipping {file_path}: Invalid JSON ({e})")
    return None
  except Exception as e:
    print(f"Skipping {file_path}: Error reading ({e})")
    return None
      
  try:
    new_path = os.path.join(logged_dir, os.path.basename(file_path))
    shutil.move(file_path, new_path)
    print(f"Moved {file_path} to {new_path}")
  except Exception as e:
    print(f"Error moving {file_path} to {logged_dir}: {e}")
  return json_data

def clean_startup(directory, logged_dir="logged"):
  for entry in os.listdir(logged_dir):
    path = os.path.join(logged_dir, entry)
    if os.path.isfile(path) or os.path.islink(path):
      os.remove(path)
  for entry in os.listdir(directory):
    if entry == logged_dir: continue
    path = os.path.join(directory, entry)
    if os.path.isfile(path) or os.path.islink(path):
      os.remove(path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--directory', type=str, default='.')
  args = parser.parse_args()
  logged_dir = str(pathlib.Path(args.directory) / "logged")
  os.makedirs(logged_dir, exist_ok=True)  

  clean_startup(args.directory)

  wandb.login()
  run = wandb.init(project="Pluribus", config={})

  print("Logging to wandb...")
  while True:
    for filename in os.listdir(args.directory):
      file_path = os.path.join(args.directory, filename)
      if os.path.isfile(file_path):
        data = read_and_move(file_path, logged_dir)
        if data: wandb.log(data)
    time.sleep(0.5)
