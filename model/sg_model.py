import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.ops as TO
import numpy as np
from .backbone import *
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from pcdet.models import build_network, load_data_to_gpu
import torchvision.models as models
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

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
    
    
class Tran_SG(nn.Module):
    def __init__(self, d_model=512, nhead=8, N=6, num_token=2, out_feature_dim=256):
        super(Tran_SG, self).__init__()
        self.num_token = num_token
        self.out_feature_dim = out_feature_dim
        
        self.Group_token = nn.Parameter(torch.randn((num_token, d_model), requires_grad=True))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        
        
        self.mlp1 = nn.Sequential(nn.Linear((num_token+1)*d_model, out_feature_dim), nn.ReLU())
        
        
        
        self.mlp_PE = nn.Sequential(nn.Linear(out_feature_dim + 4, out_feature_dim), nn.Tanh())
        self.mlp2 = nn.Sequential(nn.Linear(2*out_feature_dim, out_feature_dim),nn.ReLU(),nn.Linear(out_feature_dim,1), nn.Sigmoid())
        
        
        
        self.phi = nn.Sequential(nn.Linear((num_token+1)*d_model + 4, d_model), nn.ReLU(), nn.Linear(d_model, d_model),nn.ReLU(), nn.Linear(d_model, out_feature_dim))
        self.theta = nn.Sequential(nn.Linear((num_token+1)*d_model + 4, d_model), nn.ReLU(), nn.Linear(d_model, d_model),nn.ReLU(), nn.Linear(d_model, out_feature_dim))
    def gaussian_similarity(self, x, sigma=10.0):
        """Calculate similarity between two vectors x and y using a Gaussian function.
        
        Parameters:
            x (Tensor): First vector.
            y (Tensor): Second vector.
            sigma (float): Standard deviation of the Gaussian function.
        
        Returns:
            float: Similarity value between x and y.
        """
        # Calculate Euclidean distance
        distance = pairwise_euclidean_distance(x)
        
        # Calculate Gaussian similarity
        similarity = torch.exp(-torch.pow(distance, 2) / (2 * sigma**2))
        
        return similarity
            
    def forward(self, F, bboxes):
   

        self.Group_token = self.Group_token.to(F.device)
        src = torch.concat([self.Group_token, F], dim=0)
        output = self.transformer_encoder(src)                  # [(Ng+N), d_model]\
        
        tokens = output[:self.num_token, :]
        features = output[self.num_token:,:]
        
        tokens = tokens.flatten().unsqueeze(0)
        
        N = features.shape[0]
        
        phi = self.phi(torch.concat([tokens.repeat([N,1]), features, bboxes], dim=1))            #[N, out_feature_dim]
        theta = self.theta(torch.concat([tokens.repeat([N,1]), features, bboxes], dim=1))
        # gamma_1 = torch.mm(phi,theta.T)         ### (N,N)
        # gamma_2 = torch.mm(theta,phi.T)         ### (N,N)
        

        # sg_res = nn.Sigmoid()(gamma_1 + gamma_2)
        
        sg_res = self.gaussian_similarity(phi)
        # sg_feature1 = sg_feature.reshape([N,1,self.out_feature_dim]).repeat([1,N,1])
        # sg_feature2 = sg_feature.reshape([1,N,self.out_feature_dim]).repeat([N,1,1])
        # sg_feature = torch.concat([sg_feature1,sg_feature2], dim=-1)                            #[N,N,out_feature_dim*2]
        # sg_feature = sg_feature.reshape(-1,self.out_feature_dim*2)
        
        # sg_res = self.mlp2(sg_feature)                                                          #[N^2, 1]
        # sg_res = sg_res.reshape(N,N,1).squeeze()
        
        return sg_res



class SocialGrouping_model(nn.Module):
    def __init__(
                self,
                cfg,
                d_model=512, 
                nhead=8,
                N=6, 
                num_token=2, 
                out_feature_dim=256):
        super(SocialGrouping_model, self).__init__()
        
        self.RGB_backbone = RGB_Backbone(cfg.RGB_BACKBONE)
        self.SG_tran = Tran_SG(
                d_model, 
                nhead,
                N, 
                num_token, 
                out_feature_dim)
        
        self.W = 1280
        self.H = 720
        
    def box_normalizing(self,
                        bboxes,
                        ):

        bboxes[:,(0,2)] /= self.W
        bboxes[:,(1,3)] /= self.H

        return bboxes

    def forward(self, 
                batch):
        
        (images, bboxes, pcs, bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id, action,social_group_activity, data_dict) = batch 
        _B, _T, _C, _H, _W = images.shape
        images = images.view(_B, _C, _T, _H, _W)    # reshape image for adapt GAR backbone
        images = images
        bboxes = bboxes
    
        bboxes_list = [bboxes[i,:,:] for i in range(bboxes.shape[0])]
        rgb_feature = self.RGB_backbone(images, bboxes_list, person_id)
        rgb_feature = rgb_feature.reshape(rgb_feature.shape[0], -1)
        N = rgb_feature.shape[0]
        
        A_theta = self.SG_tran(rgb_feature, self.box_normalizing(bboxes[0,:N,:]))
        
        if not self.training:
            A_theta = A_theta.fill_diagonal_(1.)
        return A_theta
