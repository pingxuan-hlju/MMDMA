import torch
import numpy as np
from scipy.io import loadmat
ddad = torch.from_numpy(np.loadtxt('./data/drugfusimilarity.txt')) #1373
ds = torch.from_numpy(loadmat('./data/net1.mat')['interaction']) #1373*173

def calculate_sim_l(ddad, ds):
    mat = torch.cat([ddad, ds], dim=1)
    s = np.zeros((mat.shape[0], mat.shape[0]))
    result = np.zeros((mat.shape[0], mat.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            result[i, j] = np.linalg.norm(mat[i] - mat[j])
            s[i, j] = np.exp(-result[i, j] ** 2 / 2)
    return s
s = calculate_sim_l(ddad, ds)
np.savetxt('./data/drughesimilarity.txt', s)