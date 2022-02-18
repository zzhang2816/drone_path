import numpy as np

def get_dist(a,b):
    return np.sqrt(np.min(np.sum(np.power(a-b,2),axis=1)))

p1 = np.load("p1.npy")
p2 = np.load("p2.npy")
p3 = np.load("p3.npy")

print(get_dist(p1,p2))
print(get_dist(p1,p3))
print(get_dist(p3,p2))