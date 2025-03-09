import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

from config import *


"""
bounding box mapping (x,y,w,h) -> (x1,y1,x2,y2)
"""
def BBox_mapping(boxes):
    x, y, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x, y, w, h = x.unsqueeze(1), y.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    return torch.cat([x1,x2,y1,y2], dim=1)
    

"""
input : bounding box 1, 2 (x, y, w, h) format, (N(# of proposal), 4)
output : GIoU Matrix, (N, N)
"""
def Get_GIoU(boxes1, boxes2):
    boxes1 = BBox_mapping(boxes1)
    boxes2 = BBox_mapping(boxes2)

    return torchvision.ops.generalized_box_iou(boxes1, boxes2)
    


"""
Given calculated Dv, Dg matrix, This : Dv(i,j), Dg(i,j) (2-dim) -> A(i,j) (1-dim) social group matrix

"""
class Social_group_MLP(nn.Module):
    def __init__(self, cfg : Config):
        super().__init__()
        self.MLPdepth = cfg.MLPdepth
        self.MLPwidth = cfg.MLPwidth
        self.MLPactivation = cfg.MLPactivation
        
        self.model = nn.Sequential()
        for i in range(self.MLPdepth):
            self.model.add_module("fc"+str(i),nn.Linear(self.MLPwidth[i],self.MLPwidth[i+1], bias=True))
            self.model.add_module("ReLU"+str(i), self.MLPactivation[i])
    
    
    def forward(self, x):
        x = self.model(x)
        return x
        
class Act_Baseline_model(nn.Module):
    """
    Baseline model reproducing JRDB-Act detection model
    Input : RGB base person feature vector(1024), (Max_num_proposal, 1024, batch_size)
            Bounding Box for proposal, (Max_num_proposal, 4, batch_size)
    Output : Group Matrix A_theta, individual action label, group action label
    """
    def __init__(self, cfg : Config):
        self.cfg = cfg
        
        self.Max_num_proposal = cfg.max_num_proposal
        self.batch_size = cfg.batch_size
        
    def Get_GIoU_Mat(self, bboxes):
        """
        input : bounding box of each frame, (batch_size, max_num_proposal, 4(x,y,w,h) )
        output : D_G matrix in paper, (batch_size, max_num_proposal, max_num_proposal)
        """

        size = bboxes.shape[0]
        res = torch.zeros([self.batch_size, self.Max_num_proposal, self.Max_num_proposal])
        for i in range(size):
            GIoU_mat = Get_GIoU(bboxes[i,:,:], bboxes[i,:,:])
            res[i,:,:] = GIoU_mat
        return res
        
    def Get_similarity_Mat(self, batch_data):
        """
        Calculate D_v matrix in paper, similarity matrix between individual features
        input : batch_data, (batch_size, Max_num_proposals, 1024(feature dim) )
        Output : D_v matrix, (batch_size, Max_num_proposals, Max_num_proposals )
        """
        method = cfg.feature_sim
        if method == "cosine":              ### cosine similarity
            res = torch.zeros([self.batch_size, self.Max_num_proposal, self.Max_num_proposal]) 
            for i in range(batch_data.shape[0]):
                print(batch_data[i,:,:].shape)
                Sim_mat = pairwise_cosine_similarity(batch_data[i,:,:])
                
                res[i,:,:] = Sim_mat
            return res
        
                
        elif method == "euclidean":         ### euclidean similarity
            return
        return 0

    def Adj2Deg(self, A):
        """
        Adjacent matrix A -> Degree Matrix D, A : (batch_size, Max_num_proposal, Max_num_proposal)
        """
        D = torch.sum(A, dim=1)
        size = D.shape[0]
        res = torch.zeros([self.batch_size, self.Max_num_proposal, self.Max_num_proposal])
        
        for i in range(size):
            res[i,:,:] = torch.diag(D[i,:])
            
        return res
        
        
    def Adj2Lap(self, A):
        """
        Adjacent matrix A -> Laplacian matrix L (batch_size, Max_num_proposal, Max_num_proposal)
        """
        D = self.Adj2Deg(A)
        return D - A
    
    def Get_eigvec_0(self, L):
        """
        Given L:(b,n,n) matrix, return eigenvector matrix corresponding eigenvalue 0 
        input : A matrix L : (b,n,n)
        output : e : (b, n)
        """
        shape = L.shape
        n = shape[2]
        b = shape[0]
        
        res = torch.zeros([b, n])
        
        E, V = torch.linalg.eig(L)                      ### E : (b, n), V : (b, n, n)

        for i in range(b):
            for j in range(n):
                if E[i,j] == 0:
                    idx = j
                    break
            res[i,:] = V[i,:,idx]

        return res

    def Get_A_bar_mat(self, A: torch.Tensor, e:torch.Tensor):
        """
        input : Given Matrix A(b,n,n), e(b,n)
        output : A_bar matrix (b,n,n)
        """
        n = self.Max_num_proposal
        b = self.batch_size
        
        res = torch.eye(n)
        res = res.reshape([1, n, n])
        res = res.repeat([b, 1, 1])
        
        e = e.unsqueeze(2)
        
        res = res - torch.matmul(e, e.transpose(1,2))
        res = torch.matmul(A,res)
        
        return res
    
    def Loss_eig(self, L_theta:torch.Tensor, L_hat:torch.Tensor):
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        
        # Get e_hat vector which is groundtruth eigenvector corresponding to 0 eigenvalue
        e_hat = self.Get_eigvec_0(L_hat).unsqueeze(2)       # (batch_size, Max_num_proposal, 1)
        e_hat_T = e_hat.transpose(1,2)                      # (batch_size, 1, Max_num_proposal)
        L_theta_T = L_theta.transpose(1,2)
        
        L_theta_bar = self.Get_A_bar_mat(L_theta, e_hat)    # (batch_size, Max_num_proposal, Max_num_proposal)
        L_theta_bar_T = L_theta_bar.transpose(1,2)
        L_eig = torch.matmul(torch.matmul(torch.matmul(e_hat_T,L_theta_T),L_theta), e_hat) + alpha*torch.exp(-1*beta*torch.trace(torch.matmul(L_theta_bar_T, L_theta_bar))) 

        
        return L_eig
    
    def Adj_information_extractor(self, A):
        """
        for example)
        input                               output
        A =    | 0, 1, 0, 0 |      ->       [[0, 1], [2, 3]]
               | 1, 0, 0, 0 |
               | 0, 0, 0, 1 |
               | 0, 0, 1, 0 |
               
               
               edge_index?
        """
        return        
        
    
    
    def forward(self, batch_data, bboxes):
        """
        batch_data shape : (batch_size, Max_num_proposals, 1024(feature dim))
        bboxes shape : (batch_size, Max_num_proposals, 4 (x, y, h, w))
        """
        ### step 0 : Load Data
        A_hat = torch.rand([self.batch_size,self.Max_num_proposal, self.Max_num_proposal ])
        
        ### step 1 : Recognize Social Group
        Dv = self.Get_similarity_Mat(batch_data)            # (batch_size, Max_num_proposal, Max_num_proposal)
        Dg = self.Get_GIoU_Mat(bboxes)                   # (batch_size, Max_num_proposal, Max_num_proposal)
        
        batch_size = Dv.shape[0]
        
        Dv = Dv.reshape(batch_size, self.Max_num_proposal, self.Max_num_proposal, 1)
        Dg = Dg.reshape(batch_size, self.Max_num_proposal, self.Max_num_proposal, 1) # In this paper, elements of Dg is quantinized 0 or 1, far and close respectively
        
        Dvg = torch.cat((Dv, Dg), dim=3)
        Dvg = Dvg.reshape(-1,2)
        
        SG_MLP = Social_group_MLP(self.cfg)
        
        A_theta = SG_MLP(Dvg)
        A_theta = A_theta.reshape(batch_size, self.Max_num_proposal, self.Max_num_proposal)     # (batch_size, Max_num_proposal, Max_num_proposal)
        
        L_theta = self.Adj2Lap(A_theta)
        L_hat = self.Adj2Lap(A_hat)
        ### Loss function of social grouping
        BCELoss = nn.BCELoss()
        L_BCE = BCELoss(A_theta, A_hat)
        L_eig = self.Loss_eig(L_theta, L_hat)
        L_mse = []                              ### 이건 진짜 모르겠음
        
        L_G = L_BCE + L_eig + L_mse
        
        ### Make Social Group feature by Max-pool
        ### Social group notation SG (batch_size, )
        
        A_theta = torch.heaviside(A_theta, self.cfg.threshhold_step)
        
        
        
        ###
        
        
        return
        

cfg = Config("test_model")        
model = Act_Baseline_model(cfg)
A = torch.Tensor([[0, 1, 0, 0],[1, 0, 0, 0], [0,0,0,1],[0,0,1,0]])

A = torch.rand([32,10,10])
A = torch.heaviside(A, torch.tensor([0.5]))
print(A)


edge_index = A.nonzero().t().contiguous()
print(edge_index.shape)