import torch.nn as nn
import torch


class Config(object):
    def __init__(self, model_name):
        
        
        self.model_name = model_name
        
        self.feature_dim = 1024
        self.batch_size = 32

        self.max_num_proposal = 10
        self.max_num_group = 10
        
        
        # individual feature vector normalized similarity method?
        self.feature_sim = "cosine" 
        
        # Social group MLP parameter
        self.MLPdepth = 3
        self.MLPwidth = [2,32,32,1]
        self.MLPactivation = [nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
        
        # L_eig parameter
        self.alpha = 0.5
        self.beta = 0.5
        
        # step function threshhold
        self.threshhold_step = torch.tensor([0.5])