import numpy as np
from cuml import KMeans
import cudf
import argparse

def cluster(data, n_clusters):
  df = cudf.DataFrame(data)
  km = KMeans(n_clusters=n_clusters)
  km.fit(df)
  print(f"{km.labels_=}")
  print(f"{km.inertia_=}")
  return km.labels_.to_pandas().values

def cluster_batched(i, n_clusters):
  batches = [np.load(f"features_{i}_b{b}.npy") for b in range(10)]
  data = np.concatenate(batches)
  filtered = data[[i % 5 != 0 for i in range(data.shape[0])]]
  print(f"{filtered.shape=}")
  df = cudf.DataFrame(filtered)
  km = KMeans(n_clusters=n_clusters)
  km.fit(df)
  labels = np.concatencate([km.predict(batch).to_pandas().values for batch in batches])
  score = sum([km.score(batch) for batch in batches])
  print(f"{labels.shape=}")
  print(f"{labels=}")
  print(f"{score=}")
  return km.labels_.to_pandas().values


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
    