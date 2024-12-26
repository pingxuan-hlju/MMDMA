import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from main.code.tools import EarlyStopping
from main.code.model import Pred
import os
import random
import numpy as np
import torch.nn.functional as F
# from cnn import CNN

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
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def train(model, train_set, test_set,z,z1,z2, fea1,fea2, epoch, learn_rate,weight_decay, cross):
    
    optimizer = torch.optim.Adam(model.parameters(), learn_rate,weight_decay=weight_decay)
    fea1 = fea1.float().cuda() 
    fea2 = fea2.float().cuda()
    
    early_stopping = EarlyStopping(patience=80, save_path='./data/loss.pth')

    for i in range(epoch):
        model.train()
        LOSS = 0
        for x1, x2, y in train_set:
            preds, ys= [], []
            x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
            z0=model(z,z1,z2,fea1.to(device),fea2.to(device),x1,x2)
            loss=nn.CrossEntropyLoss()(z0,y)
            LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds.append(z0);ys.append(y)
        preds, ys= torch.cat(preds, dim= 0),torch.cat(ys)
        print("Cross: %d  Epoch: %d / %d Loss: %0.5f" % (cross + 1, i + 1, epoch, LOSS) + "acc:"+ str((preds.argmax(dim= 1)== ys).sum()/ len(ys)))
        torch.save(model.state_dict(), "data/loss.pth")
    # test(model, test_set, cross, z,z1,z2,fea1,fea2)
        early_stopping(LOSS, model)
        if early_stopping.flag:
            print(f'early_stopping!')
            early_stop = 1
            test(model, test_set, cross, z,z1,z2,fea1, fea2)
        # 如果到最后一轮了，保存测试结果
        if i + 1 == epoch:
            test(model, test_set, cross, z,z1,z2,fea1, fea2)

def test(model, test_set, cross, z,z1,z2,fea1, fea2):
    correct = 0
    total = 0
    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()  # 使Dropout失效
    #model.load_state_dict(torch.load('./checkpoint/best_work.pth'))
    model.load_state_dict(torch.load("data/loss.pth"))
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
        with torch.no_grad():
            z0=model(z,z1,z2,fea1.to(device),fea2.to(device),x1,x2)
            pred = z0
            a = torch.max(pred, 1)[1]
        total += y.size(0)
        correct += (a == y).sum()
        predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])

    torch.save((predall, yall), './result/fold%d' % cross) #存放每折结果和标签
    print('Test_acc: ' + str((correct / total).item()))

class MyDataset(Dataset):
    def __init__(self, tri, dm):
        self.tri = tri
        self.dm = dm

    def __getitem__(self, idx):
        x, y = self.tri[idx, :]

        label = self.dm[x][y]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]


if __name__ == "__main__":
    learn_rate = 1e-3
    weight_decay = 0
    epoch = 180
    batch = 128
    batchs = 128

    fea1,fea2, train_xy, test_xy,asso_mat,asso_mat_mask= torch.load('result/embed.pth') 

    for i in range(5):
        z= torch.load('./result/z%d' % i)
        z1= torch.load('./result/z1%d' % i)
        z2= torch.load('./result/z2%d' % i)
        net = Pred().to(device)
        train_set = DataLoader(MyDataset(train_xy[i], asso_mat), batch, shuffle=True)
        test_set = DataLoader(MyDataset(test_xy[i], asso_mat), batchs, shuffle=False)
        train(net, train_set, test_set, z,z1,z2,fea1[i],fea2[i] ,epoch, learn_rate,weight_decay, i)