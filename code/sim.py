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

mmad = torch.from_numpy(np.loadtxt('./data/microbe_microbe_similarity.txt'))  
interaction = torch.from_numpy(loadmat('./data/net1.mat')['interaction'])  
ms = interaction.T  

def calculate_sim_m(mmad, ms, sigma=1.0):
    mat = torch.cat([mmad, ms], dim=1)  # 173 x (173 + 1373)
    mat_np = mat.numpy()
    s = np.zeros((mat_np.shape[0], mat_np.shape[0]))
    for i in range(mat_np.shape[0]):
        for j in range(mat_np.shape[0]):
            dist = np.linalg.norm(mat_np[i] - mat_np[j])
            s[i, j] = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return s

sigma_value = 1.0 
s_mic = calculate_sim_m(mmad, ms, sigma=sigma_value)

np.savetxt('./data/newmicrobesimilarity.txt', s_mic)
