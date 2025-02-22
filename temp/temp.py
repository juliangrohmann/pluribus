import numpy as np;

data = np.load("clusters_r3_c200.npy")
cutoff = data.shape[0] // 2
s1 = data[:cutoff]
s2 = data[cutoff:]
np.save("clusters_r3_c200_p1.npy", s1)
np.save("clusters_r3_c200_p2.npy", s2)
print(f"{data[2000000000:2000000000+10]=}")