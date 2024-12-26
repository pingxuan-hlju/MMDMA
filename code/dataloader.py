from scipy.io import loadmat
import numpy as np
import random
import torch
import math
import os
import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

def set_seed():
    seed = 1206
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

set_seed()

#  归一化的邻接矩阵
def Regularization(adj):
    row = torch.zeros(1373)
    col = torch.zeros(173)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                row[i] += 1
                col[j] += 1
    row = torch.sqrt(row)
    col = torch.sqrt(col)
    a = torch.Tensor([1])
    ADJ = torch.zeros(size=(1373, 173))
    for m in range(adj.shape[0]):
        for n in range(adj.shape[1]):
            if adj[m][n] == 1:
                temp = row[m] * col[n]
                ADJ[m][n] = torch.div(a, temp)

    return ADJ

def laplace(fea1, fea2):
    G1=fea1
    G2=fea2
    G1=torch.where(G1 < 0.8, torch.zeros_like(G1), G1)  
    G2=torch.where(G2 < 0.8, torch.zeros_like(G2), G2)  
        
    dG1=torch.sum(G1,dim=0)**(-1/2)
    dfea1=torch.where(torch.isinf(dG1),torch.full_like(dG1,0),dG1)
    G1=dfea1[:,None]*G1*dfea1[None,:]
        
    dG2=torch.sum(G2,dim=0)**(-1/2)
    dfea2=torch.where(torch.isinf(dG2),torch.full_like(dG2,0),dG2)
    G2=dfea2[:,None]*G2*dfea2[None,:]
    return G1,G2


drug_sim = torch.from_numpy(np.loadtxt('./data/drugsimilarity.txt')) #1373
drugmic = torch.from_numpy(loadmat('./data/net1.mat')['interaction']) #1373*173

def calculate_sim_l(ddad, ds):
    mat = torch.cat([ddad, ds], dim=1)
    s = np.zeros((mat.shape[0], mat.shape[0]))
    result = np.zeros((mat.shape[0], mat.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            result[i, j] = np.linalg.norm(mat[i] - mat[j])
            s[i, j] = np.exp(-result[i, j] ** 2 / 2)
    return s
s = calculate_sim_l(drug_sim, drugmic)
np.savetxt('./data/drugheatsimilarity.txt', s)
# 读入数据
def load_data():
    
    drug_simi_fu = torch.from_numpy(np.loadtxt("./data/drugsimilarity.txt"))
    drug_simi_he=torch.from_numpy(np.loadtxt("./data/drugheatsimilarity.txt" ))
    micro_simi_mat =torch.from_numpy(np.loadtxt("./data/microbe_microbe_similarity.txt"))
    asso_mat = torch.from_numpy(loadmat("./data/net1.mat")['interaction'])
    return drug_simi_fu, drug_simi_he, micro_simi_mat, asso_mat

drug_simi_mat, drug_simi_he, micro_simi_mat, asso_mat= load_data()


def tensor_shuffle(ts, dim= 0):
    return ts[torch.randperm(ts.shape[dim])]

pos_xy = asso_mat.nonzero() 
neg_xy = tensor_shuffle((asso_mat == 0).nonzero(), dim=0)  
rand_num_4940 = torch.randperm(4940) 

neg_xy, rest_neg_xy = neg_xy[0: len(pos_xy)], neg_xy[len(pos_xy):] 
pos_neg_xy = torch.cat((pos_xy, neg_xy), dim=0)[rand_num_4940]

kflod = KFold(n_splits=5, shuffle=False)
train_xy = []
test_xy = []
asso_mat_mask =[]
fea1 = []
fea2 = []
for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_xy)):
    print(f'第{fold + 1}折')
    train_xy.append(pos_neg_xy[train_xy_idx,]) 
    test = pos_neg_xy[test_xy_idx]
    test_all = torch.cat([test, rest_neg_xy], dim=0)  
    test_xy.append(test_all)    
    # @ mask test
    asso_mat_zy = asso_mat.clone()
    for index in test: 
        if asso_mat[index[0]][index[1]] == 1:
            asso_mat_zy[index[0]][index[1]] = 0
    #asso_mat_zy = Regularization(asso_mat_zy)
    DD_DM = torch.cat([drug_simi_mat, asso_mat_zy], dim=1)
    DDN_DM = torch.cat([drug_simi_he, asso_mat_zy], dim=1)
    DM_MM = torch.cat([asso_mat_zy.T, micro_simi_mat], dim=1)
    embed1 = torch.cat([DD_DM, DM_MM], dim=0) 
    embed2 = torch.cat([DDN_DM, DM_MM], dim=0) 
    fea2.append(embed2)

torch.save([fea1,fea2, train_xy, test_xy,asso_mat,asso_mat_mask],'reslut/embed.pth')

#print(train_xy.shape)















