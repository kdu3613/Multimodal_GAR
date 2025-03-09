import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.ops as TO
import numpy as np
from .backbone import *
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from pcdet.models import build_network, load_data_to_gpu
import math




class cross_attention_fusion(nn.Module):
    def __init__(self, input_dim=512,out_dim=512):
        super(cross_attention_fusion, self).__init__()
        self.Att1 = nn.MultiheadAttention(512, 8)
        self.Att2 = nn.MultiheadAttention(512, 8)
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])

        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])

    def forward(self,R, L):
        R = self.Att1(L,R,R)[0] + R
        R = self.LN_r_1(R)
        R = self.FFN_r(R) + R
        R = self.LN_r_2(R)
        
        L = self.Att1(R,L,L)[0] + L
        L = self.LN_l_1(L)
        L = self.FFN_r(L) + L
        L = self.LN_l_2(L)
        
        res, _ = torch.max(torch.stack((R, L)), dim=0)
        return res
        
class SpaTemp_self_att(nn.Module):
    def __init__(self,in_channels, inter_channels=None, mode='dot', pool="avg"):
        super(SpaTemp_self_att,self).__init__()
        
        self.Spa_block = NLBlockND(in_channels, inter_channels, mode, dimension=2)
        
        if pool=='flat':            ### only LiDAR
            self.temp_block = NLBlockND(96 * 6 * 6, 432 ,mode, dimension=1)
        else:
            self.temp_block = NLBlockND(in_channels, inter_channels, mode, dimension=1)
            
        if pool == 'avg':
            self.pool_layer = nn.AdaptiveAvgPool2d((1))
        elif pool == 'flat':
            self.pool_layer = nn.Flatten()
    def forward(self, x):
        """

        Args:
            x (N,C,H,W): N person C channel H height W width
        """
        
        x = self.Spa_block(x)               ### x (N,C,H,W)
        
        x = self.pool_layer(x)              ### x (N,C',1)
        x = x.squeeze()                     ### x (N,C)
        x = x.unsqueeze(0)                  ### x (1, N, C)
        x = x.permute(0,2,1)                ### X (1, C, N)
        
        x = self.temp_block(x)              ### x (1, C, N)
        x = x.permute(2,1,0).squeeze()      ### x (N, C)
        
        return x
        
class FusionAttention(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
        
        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) 
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        

        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) 
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)

        return R_prime, L_prime

        
class FusionAttention2(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention2, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
        E_r = 1/(2*self.sigma**2) * torch.exp(-De**2)
        
        E_r = torch.exp(-1/(2*self.sigma**2) * (De**2))
        
        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
        
        
        
        
        
        
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) + E_l
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        
        
        
        res, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)
        return res


class FusionAttention3(nn.Module):      ### return R L
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention3, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
         
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
        E_r = 1/(2*self.sigma**2) * torch.exp(-De**2)
        
        E_r = torch.exp(-1/(2*self.sigma**2) * (De**2))
        

        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) + E_l
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        return R_prime, L_prime

class FusionAttention_gaussian(nn.Module):      ### return R L
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_gaussian, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
         
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)

        E_r = 1/(self.sigma * math.sqrt(2*math.pi)) * torch.exp((-1/2)*(De/self.sigma)**2)
        

        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        return R_prime, L_prime


class FusionAttention_mat(nn.Module):      ### return R L
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_mat, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
         
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)


        # E_r = 1/(self.sigma * math.sqrt(2*math.pi)) * torch.exp((-1/2)*(De/self.sigma)**2)
        E_r = torch.softmax(-(De/self.sigma), dim=1)

        Att_weight = (torch.matmul(Q_r, K_r.T) * E_r / self.out_dim**0.5) 
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) * E_r / self.out_dim**0.5)
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        return R_prime, L_prime







class FusionAttention_MMCA_sty(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_MMCA_sty, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim, out_dim]
        self.WQ = nn.Parameter(torch.zeros(shape))
        self.WK = nn.Parameter(torch.zeros(shape))
        self.WV = nn.Parameter(torch.zeros(shape))
        
        self.LN_1 = nn.LayerNorm([out_dim])
        self.FFN= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_2 = nn.LayerNorm([out_dim])

        
        torch.nn.init.kaiming_normal_(self.WQ) 
        torch.nn.init.kaiming_normal_(self.WK)
        torch.nn.init.kaiming_normal_(self.WV)
        

        
    def forward(self, R, L, Dg, De, Distance=False):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        F = torch.concat([R,L], dim=0)          ### (2N * D)
        
        Q = torch.matmul(F,self.WQ)
        K = torch.matmul(F,self.WK)
        V = torch.matmul(F,self.WV)

        E_r = 1/(self.sigma * math.sqrt(2*math.pi)) * torch.exp((-1/2)*(De/self.sigma)**2)
        
        E_r = nn.Sigmoid()(torch.exp(-(De/self.sigma)**2))
        
        E_r = E_r.repeat([2,2])
        
        
        if Distance:
            Att_weight = (torch.matmul(Q, K.T) / self.out_dim**0.5) * E_r
        else: 
            Att_weight = (torch.matmul(Q, K.T) / self.out_dim**0.5)
        Att_map = torch.softmax(Att_weight, dim=1)
        F_prime = torch.matmul(Att_map, V) 
        
        F_prime = self.LN_1(F_prime + F)
        F_prime = F_prime + self.FFN(F_prime) 
        F_prime = self.LN_2(F_prime)
        
        
        R_prime = F_prime[:R.shape[0],:]
        L_prime = F_prime[R.shape[0]:,:]
        

        

        return R_prime, L_prime


    
class FusionAttention_cat(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_cat, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
        E_r = 1/(2*self.sigma**2) * torch.exp(-De**2)
        
        E_r = torch.exp(-1/(2*self.sigma**2) * (De**2))
        
        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
        

        
        
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) + E_l
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        
        
        
        res = torch.concat([R_prime, L_prime], dim=1)
        return res
    
    
class FusionAttention_sum(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_sum, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, Dg, De):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            Dg (tensor) : N*N GIoU
            De (tensor) : N*N Euclid

        Returns:
            _type_: _description_
        """
        
        #### Attention R
        Q_r = torch.matmul(L,self.WQ_r)
        K_r = torch.matmul(R,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
        E_r = 1/(2*self.sigma**2) * torch.exp(-De**2)
        
        E_r = torch.exp(-1/(2*self.sigma**2) * (De**2))
        
        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
    
        #### Attention L
        Q_l = torch.matmul(R,self.WQ_l)
        K_l = torch.matmul(L,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)
        E_l = Dg
        
        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) + E_r
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        
        
        
        res = (R_prime + L_prime) / 2
        return res

    
class FusionAttention_pe(nn.Module):
    def __init__(self, input_dim=512,out_dim=512, sigma=10):
        super(FusionAttention_pe, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.sigma=sigma
        
        shape = [input_dim+2, out_dim]
        self.WQ_r = nn.Parameter(torch.zeros(shape))
        self.WK_r = nn.Parameter(torch.zeros(shape))
        shape = [input_dim, out_dim]
        self.WV_r = nn.Parameter(torch.zeros(shape))
        
        self.LN_r_1 = nn.LayerNorm([out_dim])
        self.FFN_r= nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim))
        self.LN_r_2 = nn.LayerNorm([out_dim])
  
        
        shape = [input_dim+2, out_dim]
        self.WQ_l = nn.Parameter(torch.zeros(shape))
        self.WK_l = nn.Parameter(torch.zeros(shape))
        shape = [input_dim, out_dim]
        self.WV_l = nn.Parameter(torch.zeros(shape))
        
        
        
        self.LN_l_1 = nn.LayerNorm([out_dim])
        self.FFN_l = nn.Sequential(nn.Linear(out_dim,out_dim), nn.ReLU(),nn.Linear(out_dim,out_dim) )
        self.LN_l_2 = nn.LayerNorm([out_dim])
        
        
        
        torch.nn.init.kaiming_normal_(self.WQ_r) 
        torch.nn.init.kaiming_normal_(self.WK_r)
        torch.nn.init.kaiming_normal_(self.WV_r)
        
        torch.nn.init.kaiming_normal_(self.WQ_l)
        torch.nn.init.kaiming_normal_(self.WK_l)
        torch.nn.init.kaiming_normal_(self.WV_l)
        
        
    def forward(self, R, L, bb):
        """
        Fusion method of RGB,LiDAR feature

        Args:
            R (tensor): N*D RGB_feature
            L (tensor): N*D LiDAR_feature
            bb(tensor): N*2 (cx,cy)

        Returns:
            _type_: _description_
        """
        
        #### Attention R\
            
        L_pe = torch.concat([bb, L], dim=1)
        R_pe = torch.concat([bb, R], dim=1)
        
        Q_r = torch.matmul(L_pe,self.WQ_r)
        K_r = torch.matmul(R_pe,self.WK_r)
        V_r = torch.matmul(R,self.WV_r)
 
  
        
        Att_weight = (torch.matmul(Q_r, K_r.T) / self.out_dim**0.5) 
        Att_map = torch.softmax(Att_weight, dim=1)
        R_prime = torch.matmul(Att_map, V_r) 
        
        R_prime = self.LN_r_1(R_prime + R)
        R_prime = R_prime + self.FFN_r(R_prime) 
        R_prime = self.LN_r_2(R_prime)
        
    
        #### Attention L
        Q_l = torch.matmul(R_pe,self.WQ_l)
        K_l = torch.matmul(L_pe,self.WK_l)
        V_l = torch.matmul(L,self.WV_l)

        Att_weight = (torch.matmul(Q_l, K_l.T) / self.out_dim**0.5) 
        Att_map = torch.softmax(Att_weight, dim=1)
        L_prime = torch.matmul(Att_map, V_l) 
        
        L_prime = self.LN_l_1(L_prime + L)
        L_prime = L_prime + self.FFN_l(L_prime) 
        L_prime = self.LN_l_2(L_prime)
        
        
        return R_prime, L_prime


class LiDAR_Backbone(nn.Module):
    def __init__(self, cfg, dataset):
        super(LiDAR_Backbone, self).__init__()
        self.cfg = cfg
        self.model = build_network(model_cfg=cfg.LiDAR_BACKBONE.MODEL, num_class=len(cfg.LiDAR_BACKBONE.CLASS_NAMES), dataset=dataset)
        
        
        
        
        
        if cfg.LiDAR_BACKBONE.SELF_ATT1.USE == True:
            self.self_attention_net1 = NLBlockND(96,inter_channels=96//8
                                            ,mode='dot',dimension=cfg.LiDAR_BACKBONE.SELF_ATT1.DIM)
        if cfg.LiDAR_BACKBONE.SELF_ATT1.USE == True and cfg.LiDAR_BACKBONE.SELF_ATT1.INTER_PERSON == False:
            self.embedding = nn.Linear(96*6*6*6, 512)
        elif cfg.LiDAR_BACKBONE.SELF_ATT1.USE == True and cfg.LiDAR_BACKBONE.SELF_ATT1.INTER_PERSON == True:
            self.embedding = nn.Linear(96*6*6, 512)
            
        
        if cfg.LiDAR_BACKBONE.two_stage_att and cfg.LiDAR_BACKBONE.pool == 'flat':
            self.self_attetion = SpaTemp_self_att(96, 96//8,mode='dot',pool='flat')
            self.embedding = nn.Linear(96*6*6, 512)
        elif cfg.LiDAR_BACKBONE.two_stage_att:
            self.self_attetion = SpaTemp_self_att(96, 96//8,mode='dot')
            self.embedding = nn.Linear(96, 512)
            
        
        
    def LiDAR_feature_processing(self, data_dict):
        # sharaed_feature shape : [batch_size * max_num_proposal in batch, 1024]
        # return to : [batch_size, max_num_proposals (const, ex) 100), 1024]
        MAX_NUM_PROPOSAL = self.cfg.DATALOADER.train.augmentation.num_boxes
        
        shared_feature = data_dict['shared_feature']
        batch_size = data_dict['batch_size']
        # gt_bboxes = data_dict['gt_bboxes']
        
        _N, _D = shared_feature.shape
        shared_feature = shared_feature.reshape([batch_size, -1, _D])
        
        return shared_feature
    
    def forward(self, data_dict):
        output_dict = self.model(data_dict)
        
        if self.cfg.LiDAR_BACKBONE.two_stage_att:
            shared_feature = output_dict['pooled_features']                 ### (Num_person, 6*6*6, 96)
                
            shared_feature = shared_feature.permute(0, 2, 1)

            NP, C, HWT = shared_feature.shape
            H = 6
            
            shared_feature = shared_feature.reshape(NP, C, H, H, H)         ### (Num_person, 96, 6, 6, 6)
            
            m = nn.AdaptiveAvgPool3d((6,6,1))
            
            shared_feature = m(shared_feature).squeeze(-1)                              ### (Num_person, 96, 6, 6)
            shared_feature = self.self_attetion(shared_feature)                         ### (N,96)
            shared_feature = self.embedding(shared_feature.unsqueeze(0))                ### (1, N, 96)
            
            return shared_feature
        
        if self.cfg.LiDAR_BACKBONE.SELF_ATT1.USE == False:
            shared_feature = self.LiDAR_feature_processing(output_dict)     ### (Num_person, 512)
        
        elif self.cfg.LiDAR_BACKBONE.SELF_ATT1.USE == True:
            if self.cfg.LiDAR_BACKBONE.SELF_ATT1.DIM == 3 and self.cfg.LiDAR_BACKBONE.SELF_ATT1.INTER_PERSON == False:
                shared_feature = output_dict['pooled_features']                ### (Num_person, 6*6*6 (HWT), 96(C))
                shared_feature = shared_feature.permute(0, 2, 1)

                NP, C, HWT = shared_feature.shape
                H = 6
                
                shared_feature = shared_feature.reshape(NP, C, H, H, H)         ### (Num_person, 96, 6, 6, 6)
                shared_feature = self.self_attention_net1(shared_feature)       ### (NP, 96, 6, 6, 6)

                shared_feature = shared_feature.unsqueeze(0)                    ### (1, NP, 96, 6, 6, 6)
                shared_feature = shared_feature.reshape(1, shared_feature.shape[1], -1)     ###(1, NP, 96*6*6*6)
                shared_feature = self.embedding(shared_feature)                 ### (1, NP, 512)
                
            elif self.cfg.LiDAR_BACKBONE.SELF_ATT1.DIM == 3 and self.cfg.LiDAR_BACKBONE.SELF_ATT1.INTER_PERSON == True:                    ### attention perform inter person also
                shared_feature = output_dict['pooled_features']                 ### (Num_person, 6*6*6, 96)
                
                shared_feature = shared_feature.permute(0, 2, 1)

                NP, C, HWT = shared_feature.shape
                H = 6
                
                shared_feature = shared_feature.reshape(NP, C, H, H, H)         ### (Num_person, 96, 6, 6, 6)
                
                m = nn.AdaptiveAvgPool3d((6,6,1))
                
                shared_feature = m(shared_feature).squeeze(-1)                              ### (Num_person, 96, 6, 6)
                shared_feature = shared_feature.unsqueeze(0)                                ### (1(batch_size), NP, 96, 6, 6)
                shared_feature = shared_feature.permute(0,2,1,3,4)                          ### (1(batch_size), 96(C),NP , 6, 6)
                shared_feature = self.self_attention_net1(shared_feature)
                   
                shared_feature = shared_feature.permute(0,2,1,3,4)                          ### (1(batch_size),NP,96, 6, 6)
                shared_feature = shared_feature.reshape(1, shared_feature.shape[1], -1)     ### (1, NP, 96*6*6)

                shared_feature = self.embedding(shared_feature)                             ### (1, NP, 512)
        
        return shared_feature ###shared_feature
        
class RGB_Backbone(nn.Module):
    #def __init__(self, cfg):
    def __init__(self, cfg):
        super(RGB_Backbone, self).__init__()
        """
        #self.cfg = cfg
        #T, N = self.cfg.num_frames, self.cfg.num_boxes
        #D=self.cfg.emb_features
        #K=self.cfg.crop_size[0]
        #NFB=self.cfg.num_features_boxes
        #NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        #NG=self.cfg.num_graph
        """
        self.cfg = cfg
        self.backbone_net=InceptionI3d(final_endpoint='Mixed_4f')
        self.backbone_net.build()
        #pretrained Kinetics
        pretrained_dict = torch.load('/mnt/server6_hard1/donguk/Multimodal_GAR/checkpoints/pretrained/rgb_imagenet.pt')
        self.backbone_net.load_state_dict(pretrained_dict,strict=False)
        
        
        # freeze backbone I3D
        if cfg.I3D_FREEZE:
            for para in self.backbone_net.parameters():
                para.requires_grad = False
        
        
        in_channels = 832
        
        
        if cfg.two_stage_att:
            self.self_attention_net = SpaTemp_self_att(in_channels,inter_channels=in_channels//8
                                                ,mode='dot')
        elif cfg.INTER_PERSON:
            self.self_attention_net = NLBlockND(in_channels,inter_channels=in_channels//8
                                                ,mode='dot',dimension=3)
        else:
            self.self_attention_net = NLBlockND(in_channels,inter_channels=in_channels//8
                                                ,mode='dot',dimension=2)
        
        
        
        # self.self_attention_net = NLBlockAtt(in_channels=in_channels,inter_channels=in_channels//8,key_channels=in_channels//8,query_channels=in_channels//8,val_channels=in_channels,dimension=2, mode='dot')
        self.pool_layer = nn.AdaptiveAvgPool2d((1))
        self.embedding_layer = nn.Linear(in_channels,cfg.EMBEDDING_DIM)

        self.GAT_module = pyg_nn.GATv2Conv(cfg.EMBEDDING_DIM,cfg.EMBEDDING_DIM,8,dropout=0.5,concat=False)
            
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,images_in,boxes_in, person_id):
        # read config parameters
        _B=images_in.shape[0]
        _T=images_in.shape[2]
        """
        #H, W=self.cfg.image_size
        #OH, OW=self.cfg.out_size
        #N=self.cfg.num_boxes
        #NFB=self.cfg.num_features_boxes
        #NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        #NG=self.cfg.num_graph
        
        #D=self.cfg.emb_features
        #K=self.cfg.crop_size[0]

        # Use backbone to extract features of images_in
        # Pre-precess first
        #images_in_flat=prep_images(images_in_flat)
        """
        _B = person_id.shape[0]
        person_num = [len(torch.unique(person_id[i])) - 1 for i in range(_B)]
        outputs = self.backbone_net.extract_features(images_in)

        outputs = outputs[:,:,outputs.shape[2]//2,:,:]

        W = images_in.shape[-1]
        W_f = outputs.shape[-1]
        

        boxes_features = TO.roi_align(outputs,boxes_in,output_size=5,
                                      spatial_scale=W_f/W)

        boxes_features = boxes_features[:person_num[0]]
        
        
        if self.cfg.two_stage_att:
            boxes_features = self.self_attention_net(boxes_features)                ### x (N,C)
        elif self.cfg.INTER_PERSON:
            boxes_features = boxes_features.unsqueeze(0)
            boxes_features = boxes_features.permute(0,2,1,3,4)                      ### (B, C, N(=T), H, W)
            
            
            boxes_features = self.self_attention_net(boxes_features)

            boxes_features = self.pool_layer(boxes_features)                        ### (B,C,N(=T),1,1)
            C = boxes_features.shape[1]

            boxes_features = boxes_features.squeeze().reshape(_B,C,person_num[0])
            boxes_features = boxes_features.permute(0,2,1)
        else:   
            boxes_features = self.self_attention_net(boxes_features)                ### (N, C, H, W)
            boxes_features = self.pool_layer(boxes_features)

        boxes_features = self.embedding_layer(boxes_features.squeeze())

        if self.cfg.GAT_module:
            boxes_len = [len(box) for box in boxes_in]
            boxes_len = [(box.sum(dim=1) != 0).sum().item() for box in boxes_in]
            edge_indexes = [torch.combinations(torch.arange(0,box_len) + 
                            sum(boxes_len[:max(0,i)]), r=2) 
                            for i,box_len in enumerate(boxes_len)]
            edge_index = torch.cat(edge_indexes,0)
    
            
            device = boxes_features.device
            edge_index = torch.cat((edge_index,torch.flip(edge_index,[1])),0).T.to(device)
            
            boxes_features = self.GAT_module(boxes_features,edge_index)
        return boxes_features



class Actionhead(nn.Module):
    def __init__(self, input_dim):
        super(Actionhead, self).__init__()
        self.pose_head_1 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        self.pose_head_2 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        self.pose_head_3 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        
        self.intrctn_head_1 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,2),nn.Sigmoid())
        self.intrctn_head_2 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        self.intrctn_head_3 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,7),nn.Sigmoid())
        self.intrctn_head_4 = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,5),nn.Sigmoid())
    
    def forward(self, x):
        # Softmax 1 (CE) [walking, standing, sitting, other]
        pose_1 = self.pose_head_1(x)
        # Softmax 2 (CE) [cycling, going upstairs, bending, other]
        pose_2 = self.pose_head_2(x)
        # Softmax 3 (CE) [going downstairs, skating, scootering, running, other] 
        pose_3 = self.pose_head_3(x)
        
        # Sigmoid 1 (BCE)  [whether there exists any interaction-based action]
        intrctn_1 = self.intrctn_head_1(x)
        # Sigmoid 2 (BCE) [holding sth, listening to someone, talking to someone, other]
        intrctn_2 = self.intrctn_head_2(x)
        # Sigmoid 3 (BCE)  [looking at robot, looking into sth, looking at sth, typing, interaction with door, eating sth, other]
        intrctn_3 = self.intrctn_head_3(x)
        # Sigmoid 4 (BCE) [talking on the phone, reading, pointing at sth, pushing, greeting gestures].
        intrctn_4 = self.intrctn_head_4(x)
        
        return pose_1, pose_2, pose_3, intrctn_1, intrctn_2, intrctn_3, intrctn_4

class GAR_Fusion_Net3(nn.Module):
    def __init__(self,cfg) -> None:
        super(GAR_Fusion_Net3, self).__init__()
  
        self.cfg = cfg        

        
        if cfg.EUCLIDEAN:
            self.D_embed = nn.Sequential(nn.Linear(2,1),nn.Sigmoid())   
            # self.D_embed = nn.Sequential(nn.Linear(3,8),nn.ReLU(), nn.Linear(8,4), nn.ReLU(), nn.Linear(4,1), nn.Sigmoid())
        else:   
            self.D_embed = nn.Sequential(nn.Linear(2,4),nn.ReLU(), nn.Linear(4,1),nn.Sigmoid())
        
        
        if cfg.get("Social_Layer"):
            self.social_layer = nn.Sequential(nn.Linear(int(cfg.FEATURE_DIM/2), 256), nn.ReLU(), nn.Linear(256, 128))
        if cfg.get("Social_Encoder"):
            self.social_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            
        #In each partition excluding the last one, we
        #add an “Other” class which shows the presence of an action class in the less frequent partitions.
        #our model 1
        self.pose_head_1 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        self.pose_head_2 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        self.pose_head_3 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Softmax(dim=1))
        
        self.intrctn_head_1 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,2),nn.Sigmoid())
        self.intrctn_head_2 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        self.intrctn_head_3 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,7),nn.Sigmoid())
        self.intrctn_head_4 = nn.Sequential(nn.Linear(cfg.FEATURE_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,5),nn.Sigmoid())
        
        
        ### above is individual action, below is Social group activity
        self.SG_pose_head_1 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        self.SG_pose_head_2 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        self.SG_pose_head_3 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        
        self.SG_intrctn_head_1 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,2),nn.Sigmoid())
        self.SG_intrctn_head_2 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,4),nn.Sigmoid())
        self.SG_intrctn_head_3 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,7),nn.Sigmoid())
        self.SG_intrctn_head_4 = nn.Sequential(nn.Linear(cfg.HIDDEN_DIM,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,5),nn.Sigmoid())
        if self.cfg.FUSION == "Attention_mat":
            self.AttFusModule1 = FusionAttention_mat(sigma=self.cfg.SIGMA)
            self.AttFusModule2 = FusionAttention_mat(sigma=self.cfg.SIGMA) 
        
        if self.cfg.FUSION == "Attention_multi":
            self.AttFusModule1 = FusionAttention3(sigma=3.)
            self.AttFusModule2 = FusionAttention2(sigma=1.) 

        if self.cfg.FUSION == "Attention_multi_cat":
            if self.cfg.get("Layer") == 2:
                self.AttFusModule1 = FusionAttention3(sigma=1.)
                self.AttFusModule2 = FusionAttention3(sigma=0.5)
            if self.cfg.get("Layer") == 4:
                self.AttFusModule1 = FusionAttention3(sigma=5.)
                self.AttFusModule2 = FusionAttention3(sigma=3.)
                self.AttFusModule3 = FusionAttention3(sigma=1.)
                self.AttFusModule4 = FusionAttention3(sigma=0.5)
            
        
        if self.cfg.FUSION == "Attention_normal":
            self.AttFusModule1 = FusionAttention(sigma=self.cfg.SIGMA) 
            self.AttFusModule2 = FusionAttention(sigma=self.cfg.SIGMA) 
        
        if self.cfg.FUSION == "Attention_gaussian":
            self.AttFusModule1 = FusionAttention_gaussian(sigma=3) 
            self.AttFusModule2 = FusionAttention_gaussian(sigma=3) 
            self.AttFusModule3 = FusionAttention_gaussian(sigma=3) 
            self.AttFusModule4 = FusionAttention_gaussian(sigma=3) 
        
        if self.cfg.FUSION == "Attention_max":
            
            self.AttFusModule = FusionAttention2(sigma=self.cfg.SIGMA) 
            self.phi = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
            self.sigma = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
        
        
        if self.cfg.FUSION == "Attention":
            
            self.AttFusModule = FusionAttention2(sigma=self.cfg.SIGMA) 
            self.phi = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
            self.sigma = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
            
        if self.cfg.FUSION == "Attention_sum":
            
            self.AttFusModule = FusionAttention_sum(sigma=self.cfg.SIGMA) 
            self.phi = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
            self.sigma = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
        
        
        if self.cfg.FUSION == "Attention_concat":
            
            self.AttFusModule = FusionAttention_cat(sigma=self.cfg.SIGMA)
            
            
        if self.cfg.FUSION == "Attention_pe":
            
            self.AttFusModule1 = FusionAttention_pe(sigma=self.cfg.SIGMA) 
            self.AttFusModule2 = FusionAttention_pe(sigma=self.cfg.SIGMA) 
            
        if self.cfg.FUSION == "Attention_MMCA_sty":
            self.AttFusModule1 = FusionAttention_MMCA_sty(sigma=self.cfg.SIGMA) 
            self.AttFusModule2 = FusionAttention_MMCA_sty(sigma=self.cfg.SIGMA)
        
        if self.cfg.FUSION == 'catandAtt':
            self.Att = nn.MultiheadAttention(512,8)
            self.FL = nn.Linear(1024, 512)
            self.LN = nn.LayerNorm([512])
            self.FL2 = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512,512))
            self.LN2 = nn.LayerNorm([512])
        
        if self.cfg.FUSION == "crossAtt":
            self.AttFusModule = cross_attention_fusion()
            self.D_embed = nn.Sequential(nn.Linear(32,8),nn.ReLU(), nn.Linear(8,1),nn.Sigmoid())   
            self.F_embed = nn.Linear(512, 30)

        ### fusion feature dimension
        self.f_dim = cfg.FEATURE_DIM
        self.card_net = nn.Sequential(nn.Linear(513,512), nn.ReLU(), nn.Linear(512,1))
    
        ### Each Modality Norm
        self.bn_rgb = nn.BatchNorm1d(512)
        self.bn_lidar = nn.BatchNorm1d(512)
        
        if cfg.sim == "Graph":
            self.phi = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
            self.sigma = nn.Sequential(nn.Linear(512,32),nn.ReLU(),nn.Linear(32,32))
        elif cfg.sim == "Graph2":
            # self.phi = nn.Sequential(nn.Linear(515,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))
            # self.sigma = nn.Sequential(nn.Linear(515,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,32))
            
            # self.phi = nn.Sequential(nn.Linear(515,32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32,8))
            # self.sigma = nn.Sequential(nn.Linear(515,32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32,8))
            
            # self.phi = nn.Sequential(nn.Linear(515,8),nn.ReLU(),nn.Linear(8,8))
            # self.sigma = nn.Sequential(nn.Linear(515,8),nn.ReLU(),nn.Linear(8,8))
            
            self.phi = nn.Sequential(nn.Linear(515,8))
            self.sigma = nn.Sequential(nn.Linear(515,8))
            
        elif cfg.sim == "Graph4":
            self.phi = nn.Sequential(nn.Linear(515,8))
            
    def get_f_dim(self):
        return self.f_dim
    
    def get_num_person(self, person_id):
        _B = person_id.shape[0]
        res = [len(torch.unique(person_id[i])) - 1 for i in range(_B)]
        
        return res
        
    
    def Get_similarity_Mat(self, fusion_feature, bboxe3d = None):
        """
        Calculate D_v matrix in paper, similarity matrix between individual features
        input : fusion_featuere, [NUM_PROPOSAL,D_f]
        Output : D_v matrix, (NUM_PROPOSAL, NUM_PROPOSAL )
        """

        NUM_PROPOSAL,D_f = fusion_feature.shape
        
        
        
        if self.cfg.sim == "Graph":
            phi = self.phi(fusion_feature)          ### (N,128)
            sigma = self.sigma(fusion_feature)      ### (N,128)
            
            gamma_1 = torch.mm(phi,sigma.T)         ### (N,N)
            gamma_2 = torch.mm(sigma,phi.T)         ### (N,N)
            
            return gamma_1 + gamma_2
        
        if self.cfg.sim == "Graph2":
            phi = self.phi(torch.cat((fusion_feature, bboxe3d), dim=-1))
            sigma = self.sigma(torch.cat((fusion_feature, bboxe3d), dim=-1))
            gamma_1 = torch.mm(phi,sigma.T)         ### (N,N)
            gamma_2 = torch.mm(sigma,phi.T)         ### (N,N)
            
            
            if self.training:
                return nn.Sigmoid()(gamma_1 + gamma_2)
            else:
                return nn.Sigmoid()(gamma_1 + gamma_2).fill_diagonal_(1.)
            
        if self.cfg.sim == "Graph3":
            phi = torch.cat((fusion_feature, bboxe3d), dim=-1)
            sigma = torch.cat((fusion_feature, bboxe3d), dim=-1)

            if self.training:
                return nn.Sigmoid()(torch.mm(phi,sigma.T) / phi.shape[1])
            else:
                return nn.Sigmoid()(torch.mm(phi,sigma.T) / phi.shape[1]).fill_diagonal_(1.)
            
        if self.cfg.sim == "Graph4":
            phi = self.phi(torch.cat((fusion_feature, bboxe3d), dim=-1))            ### (N,8)
            if self.training:
                return nn.Sigmoid()(torch.mm(phi, phi.T))
            else:
                return nn.Sigmoid()(torch.mm(phi, phi.T)).fill_diagonal_(1.)
        method = "cosine"
        if method == "cosine":              ### cosine similarity
            
            if self.cfg.get("Social_Layer") or self.cfg.get("Social_Encoder"):
                fusion_feature = self.social_layer(fusion_feature)
            Sim_mat = pairwise_cosine_similarity(fusion_feature, zero_diagonal=False)

            return Sim_mat
        elif method == "euclidean":         ### euclidean similarity
            return
        return 0
            
    def Get_GIoU_Mat(self, bboxes):
        """
        input : bounding box of each frame, (NUM_PROPOSAL, 4(x1,y1,x2,y2) )
        output : D_G matrix in paper, (NUM_PROPOSAL, NUM_PROPOSAL)
        """
        [NUM_PROPOSAL,D] = bboxes.shape
        res = torch.zeros([NUM_PROPOSAL, NUM_PROPOSAL])
 
        GIoU_mat = TO.generalized_box_iou(bboxes, bboxes)
  
        return GIoU_mat
    
    def forward(self, RGB_feature, LiDAR_feature, bboxes, bboxes3d, social_group_id, person_id):
        """_summary_

        Args:
            batch_data (RGB_feature, LiDAR_feature, bboxes, bboxes3d): _description_
                        RGB_feature : [_B, MAX_NUM_PROPOSAL, D_r]
                        LiDAR_feature : [_B, MAX_NUM_PROPOSAL, D_l]
                        bboxes : [_B, MAX_NUM_PROPOSAL, 4]   , (x1,y1,x2,y2)
                        bboxes3d : [_B, MAX_NUM_PROPOSAL, 7] , (x,y,z,w,h,l,rot)
        Returns:
            _type_: _description_
        """
        ### Data processing
        person_num = self.get_num_person(person_id)

        _B,MNP= person_id.shape
        if self.cfg.MODALITY == 'RGB':
            device = RGB_feature.device 
        else:
            device = LiDAR_feature.device 
        
        A_theta_list = torch.zeros([_B,MNP,MNP], requires_grad=True).to(device)
        pose_1_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        pose_2_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        pose_3_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        intrctn_1_list = torch.zeros([_B,MNP,2], requires_grad=True).to(device)
        intrctn_2_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        intrctn_3_list = torch.zeros([_B,MNP,7], requires_grad=True).to(device)
        intrctn_4_list = torch.zeros([_B,MNP,5], requires_grad=True).to(device)
        
        SG_pose_1_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        SG_pose_2_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        SG_pose_3_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        SG_intrctn_1_list = torch.zeros([_B,MNP,2], requires_grad=True).to(device)
        SG_intrctn_2_list = torch.zeros([_B,MNP,4], requires_grad=True).to(device)
        SG_intrctn_3_list = torch.zeros([_B,MNP,7], requires_grad=True).to(device)
        SG_intrctn_4_list = torch.zeros([_B,MNP,5], requires_grad=True).to(device)
        
        
        
        card_list = torch.zeros([_B,1], requires_grad=True).to(device)
        
        for b in range(_B):
            if self.cfg.MODALITY == 'RGB' or self.cfg.MODALITY == 'Multi':
                RGB_feature_b = RGB_feature[b,:person_num[b],:]
            if self.cfg.MODALITY == 'LiDAR' or self.cfg.MODALITY == 'Multi':
                LiDAR_feature_b = LiDAR_feature[b,:person_num[b],:]
            
            
            if self.cfg.FEAT_NORM:
                RGB_feature_b = self.bn_rgb(RGB_feature_b)
                LiDAR_feature_b = self.bn_lidar(LiDAR_feature_b)
            
            if self.cfg.MODALITY == 'RGB':
                fusion_feature_b = RGB_feature_b
                
            elif self.cfg.MODALITY == 'LiDAR':
                fusion_feature_b = LiDAR_feature_b
                
            elif self.cfg.MODALITY == 'Multi':
                if self.cfg.FUSION == 'sum':
                    fusion_feature_b = RGB_feature_b + LiDAR_feature_b
                elif self.cfg.FUSION == 'concat':
                    fusion_feature_b = torch.cat((RGB_feature_b,LiDAR_feature_b), dim=1 )

                elif self.cfg.FUSION == "crossAtt":
                    fusion_feature_b = self.AttFusModule(RGB_feature_b,LiDAR_feature_b)
                
            
                elif self.cfg.FUSION == 'catandAtt':
                    fusion_feature_b = torch.concat([RGB_feature_b, LiDAR_feature_b], dim=1)
                    fusion_feature_b = self.FL(fusion_feature_b)
                    fusion_feature_b_att, _ = self.Att(fusion_feature_b,fusion_feature_b,fusion_feature_b)
                    fusion_feature_b = self.LN(fusion_feature_b + fusion_feature_b_att)
                    fusion_feature_b = self.LN2(self.FL2(fusion_feature_b) + fusion_feature_b)
                    
                elif self.cfg.FUSION == 'Attention_MMCA_sty':
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    
                    Distance=False
                    if self.cfg.get("Gaussian") == True:
                        Distance=True
                    
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De,Distance)
                    R_prime, L_prime = self.AttFusModule2(R_prime, L_prime, Dg, De,Distance)
                    fusion_feature_b, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)

                elif self.cfg.FUSION in['Attention','Attention_concat', 'Attention_sum', 'Attention_max']:
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    fusion_feature_b = self.AttFusModule(RGB_feature_b, LiDAR_feature_b, Dg, De)
                
                elif self.cfg.FUSION == 'Attention_normal':
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, None, None)
                    R_prime, L_prime = self.AttFusModule2(RGB_feature_b, LiDAR_feature_b, None, None)
                    fusion_feature_b, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)
                    
                elif self.cfg.FUSION == 'Attention_gaussian':
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De)
                    R_prime, L_prime  = self.AttFusModule2(R_prime, L_prime, Dg, De)
                    R_prime, L_prime  = self.AttFusModule3(R_prime, L_prime, Dg, De)
                    R_prime, L_prime  = self.AttFusModule4(R_prime, L_prime, Dg, De)
                    fusion_feature_b, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)
                elif self.cfg.FUSION == 'Attention_mat':
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De)
                    R_prime, L_prime  = self.AttFusModule2(R_prime, L_prime, Dg, De)
                    fusion_feature_b, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)
                elif self.cfg.FUSION == 'Attention_multi_cat':
                    pass
                
                elif self.cfg.FUSION == 'Attention_multi' :
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De)
                    fusion_feature_b = self.AttFusModule2(R_prime, L_prime, Dg, De)
                elif self.cfg.FUSION == 'Attention_multi_cat' and self.cfg.get("Layer") == 2:
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De)
                    R_prime, L_prime = self.AttFusModule2(R_prime, L_prime, Dg, De)
                    fusion_feature_b = torch.cat((R_prime,L_prime), dim=1)
                elif self.cfg.FUSION == 'Attention_multi_cat' and self.cfg.get("Layer") == 4:
                    bboxes_b = bboxes[b,:person_num[b], :]
                    bboxes3d_b = bboxes3d[b,:person_num[b],:3] 
                    Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(RGB_feature_b.device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
                    De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(RGB_feature_b.device)
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, Dg, De)
                    R_prime, L_prime = self.AttFusModule2(R_prime, L_prime, Dg, De)
                    R_prime, L_prime = self.AttFusModule3(R_prime, L_prime, Dg, De)
                    R_prime, L_prime = self.AttFusModule4(R_prime, L_prime, Dg, De)
                    fusion_feature_b = torch.cat((R_prime,L_prime), dim=1)
                    
                    
                elif self.cfg.FUSION == 'Attention_pe':
                    bb = bboxes3d[b,:person_num[b],:2] 
                    R_prime, L_prime = self.AttFusModule1(RGB_feature_b, LiDAR_feature_b, bb)
                    R_prime, L_prime  = self.AttFusModule2(R_prime, L_prime, bb)
                    fusion_feature_b, _ = torch.max(torch.stack((R_prime, L_prime)), dim=0)
                    
                    
                    
                    
            bboxes_b = bboxes[b,:person_num[b], :]
            bboxes3d_b = bboxes3d[b,:person_num[b],:3] ### cx cy cz
            Dv = self.Get_similarity_Mat(fusion_feature_b, bboxes3d_b)                ### Dv             [_B, MAX_NUM_PROPOSAL, MAX_NUM_PROPOSAL]
            Dg = TO.generalized_box_iou(bboxes_b,bboxes_b).to(device)                          ### Dg             이거때문에 지금 nan문제가 뜬ㅡ는거같다...
            De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True).to(device)
            
            if self.cfg.FUSION in['Attention', 'Attention_sum']:
                phi = self.phi(fusion_feature_b)          ### (N,128)
                sigma = self.sigma(fusion_feature_b)      ### (N,128)
                
                gamma_1 = torch.mm(phi,sigma.T)         ### (N,N)
                gamma_2 = torch.mm(sigma,phi.T)         ### (N,N)
                
                A_theta = nn.Sigmoid()(gamma_1 + gamma_2)
            
            
            elif self.cfg.FUSION == "crossAtt":
                A_feature = self.F_embed(fusion_feature_b)  ### (N,30)
                A_feature_b = A_feature.unsqueeze(1).repeat([1,person_num[b],1])
                A_feature_c = A_feature.unsqueeze(0).repeat([person_num[b],1,1])
                A_feature = A_feature_b - A_feature_c       ### (N, N, 30)
                
                Dg = Dg.unsqueeze(-1)
                De = De.unsqueeze(-1)
                
                Dvge = torch.cat((A_feature,Dg,De), dim=-1)
                Dvge = Dvge.reshape(-1,32).to(fusion_feature_b.device)
                A_theta = self.D_embed(Dvge)
                A_theta = A_theta.reshape(person_num[b], person_num[b])   

            elif self.cfg.sim =="Graph2":
                A_theta = Dv
                
            elif self.cfg.sim == "Graph3":
                A_theta = Dv
            elif self.cfg.sim == "Graph4":
                A_theta = Dv    
                
            elif self.cfg.EUCLIDEAN:
                De = pairwise_euclidean_distance(bboxes3d_b, zero_diagonal=True)
                
                Dv = Dv.unsqueeze(-1)
                Dg = Dg.unsqueeze(-1)
                De = De.unsqueeze(-1)
                
                Dvge = torch.cat((Dv,Dg), dim=-1)
                Dvge = Dvge.reshape(-1,2).to(fusion_feature_b.device)
                A_theta = self.D_embed(Dvge)
                A_theta = A_theta.reshape(person_num[b], person_num[b])
            else: 
                Dv = Dv.unsqueeze(-1)
                Dg = Dg.unsqueeze(-1)
                
                Dvg = torch.cat((Dv, Dg), dim=-1)
                Dvg = Dvg.reshape(-1,2).to(fusion_feature_b.device)
                A_theta = self.D_embed(Dvg)
                A_theta = A_theta.reshape(person_num[b], person_num[b])       ### [Return Value] predicted Adjacent Matrix
                
            
            
            if self.training==False:
                A_theta = A_theta.fill_diagonal_(1.)
            ### prediction of social group id
            
            tmp = A_theta.clone().detach()
            tmp = tmp.fill_diagonal_(1.)
            SG_pred = torch.where(tmp >= 0.5, torch.ones_like(tmp), torch.zeros_like(tmp))
            SG_id_pred = []
            
            for x in SG_pred:
                SG_id_pred.append(x.nonzero()[0][0].item())
            
            
            
            ### when model predict the individual action, grouping method from prediction? or GT?
            # group_id = social_group_id[b]
            group_id = torch.tensor(SG_id_pred)
        
            if self.cfg.get("Action_concat"):
                fusion_feature_b = torch.cat((RGB_feature_b,LiDAR_feature_b), dim=1)
            
            ## Social group feature pooling
            SG_ann_set = torch.unique(group_id)
            mask = SG_ann_set >= 0
            SG_ann_set = SG_ann_set[mask]
            res_feature = torch.zeros([ fusion_feature_b.shape[0], fusion_feature_b.shape[1] * 2], requires_grad=True).to(fusion_feature_b.device)
            # for _b, b_ann in enumerate(social_group_id):
            ind_features = fusion_feature_b
            sg_features = torch.empty_like(fusion_feature_b).copy_(fusion_feature_b)
            
            for key_ann in SG_ann_set:
                idx_key_ann = torch.where(group_id == key_ann)[0]
                SG_feat = sg_features[idx_key_ann, :]
                SG_feat_pool = torch.max(SG_feat, dim=0, keepdim=True)[0]
                # SG_feat_pool = torch.mean(SG_feat, dim=0, keepdim=True)
                SG_feat_pool = SG_feat_pool.repeat(SG_feat.shape[0], 1)
                SG_feat = torch.cat([SG_feat, SG_feat_pool], dim=-1)
                res_feature[idx_key_ann, :] = SG_feat[:, :]
                sg_features[idx_key_ann, :] = SG_feat_pool

            if self.cfg.get("sg_feat_org"):
                sg_features = fusion_feature_b
            if self.cfg.get("Non_concat"):
                res_feature = fusion_feature_b
            
            if self.cfg.get("ind_action_concat"):
                if self.cfg.MODALITY == 'LiDAR':
                    res_feature = LiDAR_feature_b
                elif self.cfg.MODALITY == 'RGB':
                    res_feature = RGB_feature_b
                else:
                    res_feature =torch.cat([RGB_feature_b, LiDAR_feature_b], dim=-1)    
            
            ### action recognition
            #Divide the pose-based and interaction-ba
            # sed action classes into several disjoint partitions
            # Softmax 1 (CE) [walking, standing, sitting, other]
            pose_1 = self.pose_head_1(res_feature)
            # Softmax 2 (CE) [cycling, going upstairs, bending, other]
            pose_2 = self.pose_head_2(res_feature)
            # Softmax 3 (CE) [going downstairs, skating, scootering, running, other] 
            pose_3 = self.pose_head_3(res_feature)
            
            # Sigmoid 1 (BCE)  [whether there exists any interaction-based action]
            intrctn_1 = self.intrctn_head_1(res_feature)
            # Sigmoid 2 (BCE) [holding sth, listening to someone, talking to someone, other]
            intrctn_2 = self.intrctn_head_2(res_feature)
            # Sigmoid 3 (BCE)  [looking at robot, looking into sth, looking at sth, typing, interaction with door, eating sth, other]
            intrctn_3 = self.intrctn_head_3(res_feature)
            # Sigmoid 4 (BCE) [talking on the phone, reading, pointing at sth, pushing, greeting gestures].
            intrctn_4 = self.intrctn_head_4(res_feature)
        

            ### social group activity recognition
            SG_pose_1 = self.SG_pose_head_1(sg_features)
            # Softmax 2 (CE) [cycling, going upstairs, bending, other]
            SG_pose_2 = self.SG_pose_head_2(sg_features)
            # Softmax 3 (CE) [going downstairs, skating, scootering, running, other] 
            SG_pose_3 = self.SG_pose_head_3(sg_features)
            
            # Sigmoid 1 (BCE)  [whether there exists any interaction-based action]
            SG_intrctn_1 = self.SG_intrctn_head_1(sg_features)
            # Sigmoid 2 (BCE) [holding sth, listening to someone, talking to someone, other]
            SG_intrctn_2 = self.SG_intrctn_head_2(sg_features)
            # Sigmoid 3 (BCE)  [looking at robot, looking into sth, looking at sth, typing, interaction with door, eating sth, other]
            SG_intrctn_3 = self.SG_intrctn_head_3(sg_features)
            # Sigmoid 4 (BCE) [talking on the phone, reading, pointing at sth, pushing, greeting gestures].
            SG_intrctn_4 = self.SG_intrctn_head_4(sg_features)
        
        
            ### for MSE grouping loss
            ind_features_pool = torch.max(ind_features, dim=0, keepdim=True)[0]
            A_theta_sum = torch.sum(A_theta).reshape([1,1])
            card_feature = torch.cat((ind_features_pool, A_theta_sum), dim=1)
            card_res = self.card_net(card_feature)
            
            A_theta_list[b,:person_num[b],:person_num[b]] = A_theta
            
            pose_1_list[b,:person_num[b]] = pose_1
            pose_2_list[b,:person_num[b]] = pose_2
            pose_3_list[b,:person_num[b]] = pose_3
            
            intrctn_1_list[b,:person_num[b]] = intrctn_1
            intrctn_2_list[b,:person_num[b]] = intrctn_2
            intrctn_3_list[b,:person_num[b]] = intrctn_3
            intrctn_4_list[b,:person_num[b]] = intrctn_4
            
            SG_pose_1_list[b,:person_num[b]] = SG_pose_1
            SG_pose_2_list[b,:person_num[b]] = SG_pose_2
            SG_pose_3_list[b,:person_num[b]] = SG_pose_3
            
            SG_intrctn_1_list[b,:person_num[b]] = SG_intrctn_1
            SG_intrctn_2_list[b,:person_num[b]] = SG_intrctn_2
            SG_intrctn_3_list[b,:person_num[b]] = SG_intrctn_3
            SG_intrctn_4_list[b,:person_num[b]] = SG_intrctn_4
            
            card_list[b] = card_res

            

        return  A_theta_list, pose_1_list, pose_2_list, pose_3_list, intrctn_1_list, intrctn_2_list, intrctn_3_list, intrctn_4_list, SG_pose_1_list, SG_pose_2_list, SG_pose_3_list, SG_intrctn_1_list, SG_intrctn_2_list, SG_intrctn_3_list, SG_intrctn_4_list, card_list
        
    def getloss(self, ):
        return

class GARNet(nn.Module):
    #def __init__(self, cfg):
    def __init__(self):
        super(GARNet, self).__init__() 
        """
        #self.cfg = cfg
        #T, N = self.cfg.num_frames, self.cfg.num_boxes
        #D=self.cfg.emb_features
        #K=self.cfg.crop_size[0]
        #NFB=self.cfg.num_features_boxes
        #NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        #NG=self.cfg.num_graph
        """
        #self.backbone = GAR_Backbone(cfg)
        self.backbone = GAR_Backbone()
        self.D_embed = nn.Sequential(nn.Linear(2,1),nn.Sigmoid())
        #In each partition excluding the last one, we
        #add an “Other” class which shows the presence of an action class in the less frequent partitions.
        
        self.pose_head_1 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,4),nn.ReLU())
        self.pose_head_2 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,4),nn.ReLU())
        self.pose_head_3 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,4),nn.ReLU())
        
        self.intrctn_head_1 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,2),nn.ReLU())
        self.intrctn_head_2 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,4),nn.ReLU())
        self.intrctn_head_3 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,7),nn.ReLU())
        self.intrctn_head_4 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,5),nn.ReLU())

        #self.num_soc_head()
        #for m in self.modules():
        #    if isinstance(m,nn.Linear):
        #        nn.init.kaiming_normal_(m.weight)
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)

    def forward(self,batch_data):
        if self.train:
            images_in, boxes_in, sg_annot = batch_data
        else:
            images_in, boxes_in = batch_data
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        """
        #H, W=self.cfg.image_size
        #OH, OW=self.cfg.out_size
        #N=self.cfg.num_boxes
        #NFB=self.cfg.num_features_boxes
        #NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        #NG=self.cfg.num_graph
        
        #D=self.cfg.emb_features
        #K=self.cfg.crop_size[0]
        """
        node_features = self.backbone(images_in,boxes_in)
        boxes_len = [len(box) for box in boxes_in]
        if self.train:
            #Max-pool social group features    
            list_sg_feat = []
            list_sg_feat_pool = []
            sg_annot_set = [torch.unique(sg_ann) for sg_ann in sg_annot]
            for k,k_annot in enumerate(sg_annot):
                vid_feat = node_features[sum(boxes_len[:k]):sum(boxes_len[:k+1])]
                for key_sg in sg_annot_set[k]:
                    sg_feat = vid_feat[torch.where(k_annot==key_sg)[0],:]
                    sg_feat_pool = torch.max(sg_feat,dim=0,keepdim=True)[0]
                    sg_feat_pool = sg_feat_pool.repeat(sg_feat.shape[0],1)
                    list_sg_feat.append(sg_feat)
                    list_sg_feat_pool.append(sg_feat_pool)
            sg_feat_pool = torch.cat(list_sg_feat_pool,0)
            node_emb_features = torch.cat([node_features,sg_feat_pool],dim=-1)
        
        # Graph spectral clustering
        # ......
        
        D_G = (TO.generalized_box_iou(torch.cat(boxes_in,0)
                                      ,torch.cat(boxes_in,0))+1)/2
        D_V = torch.cdist(node_features,node_features,p=2)
        D_cat = torch.stack([D_G,D_V],dim=-1)
        D_sim = self.D_embed(D_cat).squeeze()

        graph_feat_pool = torch.max(node_features,dim=0)[0]

        #Divide the pose-based and interaction-based action classes into several disjoint partitions
        # Softmax 1 (CE) [walking, standing, sitting, other]
        pose_1 = self.pose_head_1(node_emb_features)
        # Softmax 2 (CE) [cycling, going upstairs, bending, other]
        pose_2 = self.pose_head_2(node_emb_features)
        # Softmax 3 (CE) [going downstairs, skating, scootering, running] 
        pose_3 = self.pose_head_3(node_emb_features)
        
        # Sigmoid 1 (BCE)  [whether there exists any interaction-based action]
        intrctn_1 = self.intrctn_head_1(node_emb_features)
        # Sigmoid 2 (BCE) [holding sth, listening to someone, talking to someone]
        intrctn_2 = self.intrctn_head_2(node_emb_features)
        # Sigmoid 3 (BCE)  [looking at robot, looking into sth, looking at sth, typing, interaction with door, eating sth]
        intrctn_3 = self.intrctn_head_3(node_emb_features)
        # Sigmoid 4 (BCE) [talking on the phone, reading, pointing at sth, pushing, greeting gestures].
        intrctn_4 = self.intrctn_head_4(node_emb_features)
        
        return D_sim, pose_1, pose_2, pose_3, intrctn_1, intrctn_2, intrctn_3, intrctn_4
        #graph_features = [node_features[i*boxes_len[max(0,i-1)]:box_len + i*boxes_len[max(0,i-1)],:] for i,box_len in enumerate(boxes_len)]
        #graph_features = torch.stack(graph_features,dim=0)

class GAR_Fusion_ALL(nn.Module):
    def __init__(self, cfg, dataset):
        super(GAR_Fusion_ALL, self).__init__()
        self.cfg = cfg
        self.num_boxes = cfg.DATALOADER.train.augmentation.num_boxes
        self.modality = cfg.GAR_MODEL.MODALITY
        ### RGB Backbone
        
        if self.modality == "RGB" or self.modality == "Multi":
            self.RGB_backbone = RGB_Backbone(cfg=cfg.RGB_BACKBONE)  
        
        ### LiDAR Backbone
        if self.modality == "LiDAR" or self.modality == "Multi":
            self.LiDAR_backbone = LiDAR_Backbone(cfg=cfg, dataset=dataset)
        
        ### GAR model
        self.GAR_model = GAR_Fusion_Net3(cfg=cfg.GAR_MODEL)
    
    def check(self):
        for name, param in self.GAR_model.named_parameters():
             print(name, param.requires_grad)
        return

    def forward(self, batch):
        (images, bboxes, pcs, bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id, action,social_group_activity, data_dict) = batch 
        ### RGB backbone feature extraction
        rgb_feature = None
        lidar_feature = None
        
        if self.modality == "RGB" or self.modality == "Multi":
            _B, _T, _C, _H, _W = images.shape
            images = images.view(_B, _C, _T, _H, _W)    # reshape image for adapt GAR backbone
            images = images
            bboxes = bboxes
            
            
            bboxes_list = [bboxes[i,:,:] for i in range(bboxes.shape[0])]
            rgb_feature = self.RGB_backbone(images, bboxes_list, person_id)
            rgb_feature = rgb_feature.reshape(_B,rgb_feature.shape[0], -1)
        
        #LiDAR feature extraction 
        if self.modality == "LiDAR" or self.modality == "Multi":
            load_data_to_gpu(data_dict)
            lidar_feature = self.LiDAR_backbone(data_dict)
        
        res = self.GAR_model(rgb_feature, lidar_feature, bboxes, bboxes3d, social_group_id, person_id)
        ### GAR fusion model
        
        return res
    
    
class GARNet_All(nn.Module):
    def __init__(self, cfg):
        super(GARNet_All, self).__init__()
        self.cfg = cfg
        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph

        self.backbone_net=InceptionI3d(final_endpoint='Mixed_4f')
        self.backbone_net.build()
        pretrained_dict = torch.load('../checkpoints/pretrained/rgb_imagenet.pt') #pretrained Kinetics
        self.backbone_net.load_state_dict(pretrained_dict,strict=False)
        in_channels = 832
        self.self_attention_net = NLBlockND(in_channels,inter_channels=in_channels//8,mode='dot')
        self.pool_layer = nn.AdaptiveAvgPool2d((1))
        self.embedding_layer = nn.Linear(in_channels,1024)

        self.GAT_module = pyg_nn.GATv2Conv(1024,1024,8,dropout=0.5,concat=False)

        self.D_embed = nn.Linear(2,1)

        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])

        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]

        # Reshape the input data
        #images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        #boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        #boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        #boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        #boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        #images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone_net.extract_features(images_in)
        outputs=outputs.reshape(B,T,) 
        boxes_features = TO.roi_align(outputs,boxes_in,output_size=5,spatial_scale=outputs.shape[-1]/images_in.shape[-1])

        # self attention with Non-Local Neural Networks
        boxes_features = self.pool_layer(self.self_attention_net(boxes_features))
        boxes_features = self.embedding_layer(boxes_features.squeeze())

        boxes_len = [len(box) for box in boxes_in]
        edge_indexes = [torch.combinations(torch.arange(0,box_len) + i*boxes_len[max(0,i-1)], r=2) for i,box_len in enumerate(boxes_len)]
        edge_index = torch.cat(edge_indexes,0)
        edge_index = torch.cat((edge_index,torch.flip(edge_index,[1])),0).T
        #edge_index = torch.combinations(torch.arange(0,3), r=2)
        #edge_index = torch.cat((edge_index,torch.flip(edge_index,[1])),0)
        node_features = self.GAT_module(node_features,edge_index)
        
        # Max-pool on same social group


        # Calculate D_G
        D_G = (TO.generalized_box_iou(torch.cat(boxes_in,0),torch.cat(boxes_in,0))+1)/2
        # Calculate D_V
        D_V = torch.cdist(node_features,node_features,p=2)
        D_cat = torch.stack([D_G,D_V],dim=-1)
        D_sim = self.D_embed(D_cat).squeeze()
        
        return D_sim
        #graph_features = [node_features[i*boxes_len[max(0,i-1)]:box_len + i*boxes_len[max(0,i-1)],:] for i,box_len in enumerate(boxes_len)]
        #graph_features = torch.stack(graph_features,dim=0)

