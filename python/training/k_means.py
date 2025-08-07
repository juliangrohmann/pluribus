import numpy as np
import argparse
import time
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


def cluster(data, n_clusters):
  print(f"{data.shape=}")
  km = KMeans(
    n_clusters=n_clusters,
    init='k-means++',       # better centroid initialization
    n_init=1,               # try n different centroid seeds and pick the best
    max_iter=1000,          # allow convergence for tough clusters
    tol=1e-6,               # tighter convergence threshold
    copy_x=False,
    algorithm='lloyd',      # default; 'elkan' is faster for dense Euclidean data, but 'lloyd' is more robust for high-dimensional or sparse input
    random_state=42         # ensures determinism across runs
  )
  t_0 = time.time()
  labels = km.fit_predict(data)
  print(f"clustered in {time.time() - t_0} s")
  print(f"{labels=}")  
  print(f"{km.score(data)=}")
  return labels, km.cluster_centers_

def cluster_batched(i, n_clusters):
  # km = MiniBatchKMeans(n_clusters=n_clusters, max_iter=300, tol=1e-4)
  km = MiniBatchKMeans( # TODO: untested, compare score to previous hyperparams
    n_clusters=n_clusters,
    init='k-means++',
    max_iter=2000,              # number of mini-batches to process
    batch_size=100_000,         # large enough for stable updates, small enough for memory
    n_init=20,                  # fewer initializations since partial_fit is used
    max_no_improvement=50,      # allow more patience for convergence
    reassignment_ratio=1e-6,    # discourage centroid resets
    tol=1e-6,                   # tight convergence
    random_state=42
  )
  batches = []
  for b in range(2):
    print(f"{b=}")
    batch = np.concatenate([np.load(f"features_r{i}_b{b*5 + o}.npy") for o in range(5)])
    km.partial_fit(batch)
    batches.append(batch)
  labels = np.concatenate([km.predict(batch) for batch in batches])
  print(f"{labels=}")
  print(f"{labels.shape=}")
  print(f"{sum([km.score(np.load(f'features_r{i}_b{b}.npy')) for b in range(10)])=}")
  return labels

def to_output_fn(fn_, prefix, args_):
  return str(Path(args_.out) / (fn_.replace("features_", prefix).rstrip(".npy") + f"_c{args_.clusters}.npy"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("round", type=str)
  parser.add_argument("--clusters", type=int, default=200)
  parser.add_argument("--flops", action="store_true")
  parser.add_argument("--src", type=str, default="")
  parser.add_argument("--out", type=str, default="")
  args = parser.parse_args()
  print(vars(args))
  for r in [2, 3] if args.round == "all" else [int(args.round)]:
    files = [f"features_r{r}_f{flop_idx}.npy" for flop_idx in range(1755)] if args.flops else [f"features_{r}.npy"]
    for fn in files:
      labels, centroids = cluster_batched(r, args.clusters) if r == 3 and not args.flops else cluster(np.load(Path(args.src) / fn), args.clusters)
      np.save(clusters_fn := to_output_fn(fn, "clusters_", args), labels)
      print(f"clusters written to {clusters_fn}")
      if args.flops:
        np.save(centroids_fn := to_output_fn(fn, "centroids_", args), centroids)
        print(f"centroids written to {centroids_fn}")
