import numpy as np
from cuml import KMeans
import cudf
import argparse

def cluster(data, n_clusters):
  df = cudf.DataFrame(data)
  km = KMeans(n_clusters=n_clusters)
  km.fit(df)
  return km.labels_.to_pandas().values

def cluster_batched(i, n_clusters):
  data = np.concatenate([np.load(f"features_{i}_b{b}.npy").astype(np.half) for b in range(10)])
  print(data.shape)
  print(data)
  cluster(data, n_clusters)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("round", type=int)
  parser.add_argument("--clusters", type=int, default=200)
  args = parser.parse_args()
  print(f"{args.round=}")
  print(f"{args.clusters=}")
  labels = cluster_batched(args.round, args.clusters) if args.round == 3 else cluster(np.load(f"features_{args.round}.npy"), args.clusters)
  np.save(fn := f"clusters_r{args.round}_c{args.clusters}.npy", labels)
  print(f"clusters written to {fn}")
    