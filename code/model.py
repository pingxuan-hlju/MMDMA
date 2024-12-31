import torch   
import torch.nn as nn   
from torch.utils.data import DataLoader,Dataset
torch.cuda.empty_cache()
import torch.nn.functional as F
import random
import numpy as np
import argparse
import os
import time
from tools import EarlyStopping
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

def laplace(fea1):
    G1=fea1
    G1=torch.where(G1 < 0.8, torch.zeros_like(G1), G1)    
    dG1=torch.sum(G1,dim=0)**(-1/2)
    dfea1=torch.where(torch.isinf(dG1),torch.full_like(dG1,0),dG1)
    G1=dfea1[:,None]*G1*dfea1[None,:]
    return G1

#Multi-view Variational Graph Autoencoder
 
class Neg(nn.Module):  
    def __init__(self,params):  
        super(Neg, self).__init__()
        self.params=params
        self.conv = nn.Conv2d(params.input_dim,params.input_dim, kernel_size=(params.topk,1), stride=1, padding=0) 
        self.relu = nn.ReLU()
        self.reset_parameters()
    
    def reset_parameters(self):    
    # 初始化权重，偏置等参数    
        for m in self.modules():    
            if isinstance(m, nn.Conv2d):    
                nn.init.kaiming_uniform_(m.weight) 
            elif isinstance(m, nn.Linear):    
                nn.init.xavier_uniform_(m.weight,  gain=nn.init.calculate_gain('relu'))
    
    def forward(self, f1):    
        i = torch.topk(f1,params.topk).indices
        pos = f1[i]
        repeated_f1 = f1.repeat_interleave(params.topk, dim=0).reshape(params.input_dim,params.topk,-1) 
        out=torch.cat((repeated_f1,pos),dim=2).reshape(64,params.input_dim,params.topk,-1)
        return self.relu(self.conv(out)).reshape(params.input_dim,-1)
            
class Encode(nn.Module):  
    def __init__(self,params):  
        super(Encode, self).__init__()
        self.params=params
        self.l0_1 = nn.Linear(3*params.hidden_dim,params.hidden_dim)
        self.l0_2 = nn.Linear(3*params.hidden_dim,params.hidden_dim)
        
        self.l1_1 = nn.Linear(params.hidden_dim,params.hidden_dim)
        self.l1_2 = nn.Linear(params.hidden_dim,params.hidden_dim)
        
        self.l2_1=nn.Linear(params.hidden_dim, params.output_dim)
        self.l2_2=nn.Linear(params.hidden_dim, params.output_dim)
        self.l2_3=nn.Linear(params.hidden_dim, params.output_dim)
        self.l2_4=nn.Linear(params.hidden_dim, params.output_dim)

        self.weight1 = nn.Parameter(torch.Tensor(1546, 1546))
        self.weight2 = nn.Parameter(torch.Tensor(1546, 1546))
        self.neg1 = Neg(params)
        self.neg2 = Neg(params)

        self.relu = nn.ReLU()
        self.reset_parameters()  

    def reset_parameters(self):      
        for m in self.modules():    
            if isinstance(m, nn.Linear):    
                nn.init.xavier_uniform_(m.weight,  gain=nn.init.calculate_gain('relu'))
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
                      
    def forward(self, f1,f2,G1,G2):  
        
        xd = self.neg1(f1) 
        xd = torch.cat((f1,xd),dim=1)
        hidden10 = self.l0_1(xd)
        hidden1_0 = self.relu(torch.mm(G1, self.l1_1(hidden10)))
        
        xc = self.neg2(f2)
        xc = torch.cat((f2,xc),dim=1)
        hidden20 = self.l0_2(xc)
        hidden2_0 = self.relu(torch.mm(G2, self.l1_2(hidden20)))
        
        hidden1 = hidden1_0+(self.weight1).matmul(hidden2_0)+f1
        hidden2 = hidden2_0+(self.weight2).matmul(hidden1_0)+f2


        mean1 = self.relu(torch.mm(G1, self.l2_1(hidden1)))
        logvar1 = self.relu(torch.mm(G1, self.l2_2(hidden1)))
        
        mean2 = self.relu(torch.mm(G2, self.l2_3(hidden2)))
        logvar2 = self.relu(torch.mm(G2, self.l2_4(hidden2)))
        
        return  mean1,logvar1,hidden1,mean2,logvar2,hidden2 
    
class MultiVGAE(nn.Module):  
    def __init__(self,params):  
        super(MultiVGAE, self).__init__()
        #降维
        self.params = params
        self.l0_1 =nn.Linear(self.params.input_dim,params.hidden_dim)
        self.l0_2 =nn.Linear(self.params.input_dim,params.hidden_dim)
        #enco
        self.enco1 = Encode(params)
        #recon
        self.l1_1 = nn.Linear(params.output_dim,1024)
        self.l1_2 = nn.Linear(1024,params.input_dim)
        self.l2_1 = nn.Linear(params.output_dim,1024)
        self.l2_2 = nn.Linear(1024,params.input_dim)
        self.relu = nn.ReLU()
        self.reset_parameters()
   
    def reset_parameters(self):      
        for m in self.modules():    
            if isinstance(m, nn.Linear):    
                nn.init.xavier_uniform_(m.weight,  gain=nn.init.calculate_gain('relu'))
  
    def reparameterize1(self, mean1, logvar1):
        
        std1 = torch.exp(0.5 * logvar1)  
        eps1 = torch.randn_like(std1)  
        z1 = mean1 + eps1 * std1  
        return z1
    
    def reparameterize2(self, mean2, logvar2):
        
        std2 = torch.exp(0.5 * logvar2)  
        eps2 = torch.randn_like(std2)  
        z2 = mean2 + eps2 * std2  
        return z2

    def forward(self,f1,f2,G1,G2):
        #降维
        f1n=F.normalize(f1)
        f2n=F.normalize(f2)
        x0_1 = self.relu(self.l0_1(f1n)) 
        x0_2 = self.relu(self.l0_2(f2n)) 
       
        #encode
        mean1,logvar1,hidden1,mean2,logvar2,hidden2= self.enco1(x0_1,x0_2,G1,G2)  
    
        #reparater
        z1= self.reparameterize1(mean1, logvar1)
        z2= self.reparameterize2(mean2, logvar2)

        #decode
        out1 = self.relu(self.l1_1(z1)) 
        out1 = self.l1_2(out1)
        
        out2 = self.relu(self.l2_1(z2)) 
        out2 = self.l2_2(out2)
        
        #pre 
        zl_1=torch.cat((hidden1,z1),dim=1)
        zl_2=torch.cat((hidden2,z2),dim=1)      
        return out1, mean1,logvar1,out2,mean2,logvar2,zl_1,zl_2
     
def loss2(out1,f1, mean1, logvar1,out2,f2,mean2, logvar2):
    recon_loss1 = nn.MSELoss()(out1, f1)
    recon_loss2 = nn.MSELoss()(out2, f2)
    recon = recon_loss1+recon_loss2
    
    kld_loss1= -0.5/ 1546* (1 + 2*logvar1 - mean1**2 - torch.exp(logvar1)**2).sum(1).mean()
    kld_loss2= -0.5/ 1546* (1 + 2*logvar2 - mean2**2 - torch.exp(logvar2)**2).sum(1).mean()
    kl=kld_loss1+kld_loss2
    
    
    loss = recon+ +0.001*kl
    
    return loss
    
def trmuvg(net, params,train_set, test_set, fea1,fea2,cross):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, weight_decay=0)
    fea1 = fea1.float().cuda() 
    fea2 = fea2.float().cuda()
    G1= laplace(fea1)
    G2= laplace(fea2)
    early_stopping = EarlyStopping(patience=40, save_path="result/vg1.pth")
    for i in range(params.epoch):
        # train
        net.train()
        aloss=0
        preds, ys= [], []
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
            out1, mean1,logvar1,out2,mean2,logvar2,zl_1,zl_2=net(fea1.to(device),fea2.to(device),G1.to(device),G2.to(device))
            loss=loss2(out1,fea1.to(device), mean1, logvar1,out2,fea2.to(device),mean2, logvar2)
            aloss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            aloss+=loss
        print("Cross: %d  Epoch: %d / %d Loss: %0.5f" % (cross + 1, i + 1, params.epoch, aloss))
        torch.save(net.state_dict(), "result/vg1.pth")
        # test
    # temuvg(net,params, test_set,cross, fea1,fea2,G1,G2)
        early_stopping(loss, net)
        if early_stopping.flag:
            print(f'early_stopping!')
            early_stop = 1
            temuvg(net,params, test_set, cross, fea1,fea2,G1,G2)
        if i + 1 == params.epoch:
            temuvg(net,params, test_set, cross, fea1,fea2,G1,G2)    

def temuvg(net,params, test_set, cross, fea1, fea2,G1,G2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0

    predall, yall = torch.tensor([]), torch.tensor([])
    net.eval()  

    net.load_state_dict(torch.load("result/vg1.pth"))
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
        with torch.no_grad():
            out1, mean1,logvar1,out2,mean2,logvar2,zl_1,zl_2=net(fea1.to(device),fea2.to(device),G1.to(device),G2.to(device))
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])
    torch.save((predall, yall), './result/V%d' % cross)
    torch.save(zl_1, './result/zl_1%d' % cross)
    torch.save(zl_2, './result/zl_2%d' % cross) 
#Multi-scale Hypergraph    
class HConstructor(nn.Module):
    def __init__(self,params):
        super().__init__()
        
        self.paras=params
        self.num_edges = params.num_edges
 
        self.edges_mud = nn.Parameter(torch.randn(1, params.output_dim))#1*de
        self.edges_logsigmad = nn.Parameter(torch.zeros(1, params.output_dim))
        
        self.edges_muc = nn.Parameter(torch.randn(1, params.output_dim))#1*de
        self.edges_logsigmac = nn.Parameter(torch.zeros(1, params.output_dim))
       
        self.to_q,self.to_k,self.to_v = (nn.Linear(2*params.output_dim, 2*params.output_dim),
        nn.Linear(2*params.output_dim, 2*params.output_dim),nn.Linear(2*params.output_dim, 2*params.output_dim))
        self.relu= nn.ReLU()
        self.init_para()

    def init_para(self):
        nn.init.xavier_uniform_(self.edges_logsigmad)
        nn.init.xavier_uniform_(self.edges_logsigmac)
        nn.init.xavier_normal_(self.to_q.weight)
        nn.init.xavier_normal_(self.to_k.weight,nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.to_v.weight,nn.init.calculate_gain('relu'))
                
    def forward(self, x):
        device = x.device
       
        n_s = self.num_edges#超边数量
        
        mud = self.edges_mud.expand(n_s, -1) 
        sigmad = self.edges_logsigmad.exp().expand(n_s, -1)
        edgesd =0.001*(mud + sigmad * torch.randn(mud.shape,device=device))
        
        muc = self.edges_muc.expand(n_s, -1) 
        sigmac = self.edges_logsigmac.exp().expand(n_s, -1)
        edgesc =0.001*(muc + sigmac * torch.randn(muc.shape,device=device))
        
        edges = torch.cat((edgesd,edgesc),dim=1)

        k = self.relu(self.to_k(x))
        v = self.relu(self.to_v(x))
        q = self.to_q(edges)
        
        dots = torch.einsum('ni,ij->nj', q, k.T) #* self.scale
        attn = torch.softmax(dots, dim=1)  

        edges = torch.einsum('in,nf->if', attn, v)
    
        H = torch.einsum('ni,ij->nj',self.relu(self.to_q(x)), self.to_k(edges).T) #* self.scale 
   
        return edges,H
    
class MSHconv(nn.Module):
    def __init__(self,params):
        super(MSHconv, self).__init__()
        
        self.params=params
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.l1 = nn.Linear(2*params.output_dim,2*params.output_dim)
        self.weight1 = nn.Parameter(torch.randn(params.input_dim,params.input_dim))
        self.weight2 = nn.Parameter(torch.randn(params.input_dim,params.input_dim))
        self.l2=nn.Linear(512, 2*params.output_dim)
        self.relu = nn.ReLU()
        self.init_para()

    def init_para(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.l1.weight,nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.l2.weight)

    def forward(self,e,H,A2,x):
        #node
        n = H.matmul(self.l1(e))
        n1 = A2.matmul(e)
        n2=(self.weight1).matmul(n)+(self.weight2).matmul(n1)
        return self.leakyrelu(self.l2(torch.cat((x,n2),dim=1)))
    
class HGNN(nn.Module):
    def __init__(self,params):
        super(HGNN,self).__init__()
        
        self.params=params
        self.relu=nn.ReLU()
        self.HConstructor1 = HConstructor(params) 
        self.l1 =nn.Linear(params.input_dim,128)               
        self.convs1=MSHconv(params)
        self.convs2=MSHconv(params)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(2, 10), stride=1, padding=0),
                                   nn.MaxPool2d(kernel_size=(1, 10)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, kernel_size=(
                                       1, 10), stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 10)))
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(
            32 * 32, 300), nn.ReLU(), nn.Dropout(0.5), nn.Linear(300, 2))
        self.init_para()
        
    def init_para(self):
        nn.init.xavier_uniform_(self.l1.weight,nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(
            self.mlp[1].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp[4].weight)
        
    def forward(self,fea1,fea2,x1,x2):
        
        fea1n=F.normalize(fea1)
        fea2n=F.normalize(fea2)
        
        x0d = self.relu(self.l1(fea1n)) 
        x0c = self.relu(self.l1(fea2n))
        x0 = torch.cat((x0d,x0c),dim=1)
       
        e, H= self.HConstructor1(x0)
        
        de=torch.sum(H,dim=0)**(-1)
        le=torch.where(torch.isinf(de),torch.full_like(de,0),de)
        De=le[:,None]*H.T
        
        A2 = H.matmul(De)
        
        A2 =laplace(A2)
        A2 =A2.matmul(H)
       
        z =self.convs1(e,H,A2,x0)
        z_pre =self.convs2(e,H,A2,z)
        
        clas = torch.cat((z_pre,fea1,fea2),dim=1)
        clas=torch.cat([clas[x1],clas[x2]],dim=1)
        clas =clas.reshape(clas.shape[0],1,2,-1)
        z0=self.mlp(self.conv1(clas).view(clas.shape[0], -1))
        
        return z_pre,z0
    
def trmshy(net, params,train_set, test_set, fea1,fea2,cross):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    fea1 = fea1.float().cuda() 
    fea2 = fea2.float().cuda()
    early_stopping = EarlyStopping(patience=80, save_path="result/th.pth")
    for i in range(params.epoch):
        # train
        net.train()
        aloss= 0
        preds, ys= [], []
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
            z,z0=net(fea1.to(device),fea2.to(device),x1,x2)
            loss=nn.CrossEntropyLoss()(z0,y)
            aloss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds.append(z0);ys.append(y)
        preds, ys= torch.cat(preds, dim= 0),torch.cat(ys)
        print("Cross: %d  Epoch: %d / %d Loss: %0.5f" % (cross + 1, i + 1, params.epoch, aloss)+ "acc:"+ str((preds.argmax(dim= 1)== ys).sum()/ len(ys)))  
        torch.save(net.state_dict(), "result/th.pth")
        # test
    # temshy(net,params, test_set, cross, fea1,fea2)
        early_stopping(loss, net)
        if early_stopping.flag:
            print(f'early_stopping!')
            early_stop = 1
            temshy(net, params,test_set, cross, fea1,fea2)
        if i + 1 == params.epoch:
            temshy(net,params, test_set, cross, fea1,fea2)
def temshy(net,params, test_set, cross, fea1, fea2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    predall, yall = torch.tensor([]), torch.tensor([])
    net.eval()  
    net.load_state_dict(torch.load("result/th.pth"))
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), (x2+1373).long().to(device), y.long().to(device)
        with torch.no_grad():
            z,z0=net(fea1.to(device),fea2.to(device),x1,x2)
            pred = z0
            a = torch.max(pred, 1)[1]
        total += y.size(0)
        correct += (a == y).sum()
        predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])

    torch.save((predall, yall), './result/HY_%d' % cross)
    torch.save(z, './result/zpre%d' % cross)
    print('Test_acc: ' + str((correct / total).item()))
 
#Graphinfo   
class Graphinfo(nn.Module):
    def __init__(self,params,z,z1,z2):
        super(Graphinfo,self).__init__()
        self.params=params
        self.z = nn.Parameter(z)
        self.z1 = nn.Parameter(z1)
        self.z2 = nn.Parameter(z2)
        
    def infoloss(self,graph1,graph2):     
        graph1 = F.normalize(graph1)
        graph2 = F.normalize(graph2)
        sim_matrix  = torch.mm(graph1,graph2.t()+1e-5)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos = sim_matrix.diag()
        closs = pos/ (sim_matrix.sum(dim =1)-pos)
        closs = -torch.log(closs)
        closs = closs.mean()
        return closs

    def forward(self,z,z1,z2):   
        z =self.z
        z1 = self.z1
        z2 = self.z2
        closs1 = self.infoloss(z1,z)
        closs2 = self.infoloss(z2,z) 
        closs3 = self.infoloss(z1,z2)
        clos = closs1+closs2+closs3
   
        return z,z1,z2,clos
    
def train(model, z,z1,z2,params):
    optimizer = torch.optim.Adam(model.parameters(), params.lr)
    for i in range(params.epoch):
        time_start = time.time()
        z,z1,z2,loss = model(z,z1,z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end = time.time()
        
        print(f'epoch: {i+ 1}, loss:{loss}, time: {time_end- time_start}')

class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(2, 10), stride=1, padding=0),
                                   nn.MaxPool2d(kernel_size=(1, 10)),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, kernel_size=(
                                       1, 10), stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 10)))
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(
            32 * 32, 300), nn.ReLU(), nn.Dropout(0.5), nn.Linear(300, 2))
        self.init_para()

    def init_para(self):
        nn.init.xavier_normal_(
            self.mlp[1].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp[4].weight)

    def forward(self,z,z1,z2,f1,f2,x1,x2):
        clas1 = torch.cat([z,f1,f2],dim=1)
        clas2 = torch.cat([z1,f1,f2],dim=1)
        clas3 = torch.cat([z2,f1,f2],dim=1)
        clas=torch.cat([clas1,clas2,clas3],dim=1)
        clas=torch.cat([clas[x1],clas[x2]],dim=1)
        x= clas.reshape(clas.shape[0],3,2,-1)
        return self.mlp(self.conv1(x).view(x.shape[0], -1))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for Arguments')
    parser.add_argument('-batch', type=int, default=128)
    parser.add_argument('-device', type=str, default='cuda:0')  # cuda:0
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-temperature', type=float, default=0.1)
    parser.add_argument('-patience', type=int, default=80) 
    parser.add_argument('-epoch', type=int, default=150)
    parser.add_argument('-input_dim', type=int, default=1546)
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-output_dim', type=int, default=128)
    parser.add_argument('-num_edges', type=int, default=256)
    parser.add_argument('-topk', type=int, default=10)
    params = parser.parse_args([])
#1
    fea1,fea2, train_xy, test_xy,asso_mat,asso_mat_mask = torch.load('./embed.pth')
    for i in range(5):
        net = MultiVGAE(params).to(params.device)
        train_set = DataLoader(MyDataset(train_xy[i], asso_mat), params.batch, shuffle=True)
        test_set = DataLoader(MyDataset(test_xy[i], asso_mat), params.batch, shuffle=False)
        trmuvg(net, params, train_set, test_set, fea1[i],fea2[i],i)
# #2    
#     fea1,fea2, train_xy, test_xy,asso_mat,asso_mat_mask = torch.load('./embed.pth')
#     for i in range(5):
#         net1 = HGNN(params).to(params.device)
#         train_set = DataLoader(MyDataset(train_xy[i], asso_mat), params.batch, shuffle=True)
#         test_set = DataLoader(MyDataset(test_xy[i], asso_mat), params.batch, shuffle=False)
#         trmshy(net1, params, train_set, test_set, fea1[i],fea2[i],i)
# #3          
#     for i in range(5):
#         z= torch.load('./result/zpre%d' % i)
#         z1= torch.load('./result/zl_1%d' % i)
#         z2= torch.load('./result/zl_2%d' % i)
#         net2 = Graphinfo(params,z,z1,z2).to(params.device)
#         train (net2,z,z1,z2,params)
#         torch.save(z, './result/z%d' % i)
#         torch.save(z1, './result/z1%d' % i)
#         torch.save(z2, './result/z2%d' % i)

