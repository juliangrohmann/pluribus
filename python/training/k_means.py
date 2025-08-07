import numpy as np
import argparse
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


def cluster(data, n_clusters):
  print(f"{data.shape=}")
  km = KMeans(n_clusters=n_clusters, max_iter=300, tol=1e-4, copy_x=False)
  labels = km.fit_predict(data)
  print(f"{labels=}")  
  print(f"{km.score(data)=}")
  return labels

def cluster_batched(i, n_clusters):
  km = MiniBatchKMeans(n_clusters=n_clusters, max_iter=300, tol=1e-4)
  batches = []
  for b in range(2):
    print(f"{b=}")
    batch = np.concatenate([np.load(f"features_{i}_b{b*5 + o}.npy") for o in range(5)])
    km.partial_fit(batch)
    batches.append(batch)
  labels = np.concatenate([km.predict(batch) for batch in batches])
  print(f"{labels=}")
  print(f"{labels.shape=}")
  print(f"{sum([km.score(np.load(f'features_{i}_b{b}.npy')) for b in range(10)])=}")
  return labels

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("round", type=int)
  parser.add_argument("--clusters", type=int, default=200)
  parser.add_argument("--flops", action="store_true")
  parser.add_argument("--dir", type=str, default="")
  args = parser.parse_args()
  print(f"{args.round=}, {args.clusters=}, {args.batched}, {args.flops}, {args.dir}")
  files = [f"features_r{args.round}_f{flop_idx}.npy" for flop_idx in range(1755)] if args.flops else [f"features_{args.round}.npy"]
  for f in files:
    labels = cluster_batched(args.round, args.clusters) if args.round == 3 and not args.flops else cluster(np.load(f), args.clusters)
    np.save(fn := str(Path(args.dir) / f"clusters_r{args.round}_c{args.clusters}.npy"), labels)
    print(f"clusters written to {fn}")
