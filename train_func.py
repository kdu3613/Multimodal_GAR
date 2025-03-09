
import os
from pcdet.config import cfg, cfg_from_yaml_file

# Configuration 파일 로드
cfg_file = '/mnt/server6_hard1/donguk/Multimodal_GAR/Multimodal_cfg/mil3.yaml'
cfg_from_yaml_file(cfg_file, cfg)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import yaml
from easydict import EasyDict
from data.utils.utils import * 
from dataloader import JRDB_act
from torch.utils.data import DataLoader
from model.gat_model import *
import torch.nn as nn

import torch.optim as optim
import time
from tqdm import tqdm
from train_utils import *
from train_utils import Timer
import neptune


import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval')
import JRDB_eval as eval
from JRDB_eval_tool import analysis_result 
from make_result import *

# helper
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

# torch.cuda.set_stream(torch.cuda.Stream(0))
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


task = "task_3"
gt_file_path = {
    "group" : '/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/out/gt_group.txt',
    "action" : '/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/out/gt_action.txt',
    "activity" : '/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/out/gt_activity.txt'
}
labelmap_path = {
    "task_1" : "/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/label_map/task_1.pbtxt",
    "task_2" : "/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/label_map/task_2.pbtxt",
    "task_3" : "/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/label_map/task_3.pbtxt",
    "task_4" : "/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/label_map/task_4.pbtxt",
    "task_5" : "/mnt/server6_hard1/donguk/jrdb_toolkit/ActionSocial_grouping_eval/label_map/task_5.pbtxt"
}




def train_net(dataloader, model, optimizer, epoch, Loss, run):
    ### forwarding ###

    valset = JRDB_act(cfg.DATALOADER.train.augmentation,
                       root_path='/mnt/server14_hard1/jiny/GAR/Datasets/JRDB/',
                       is_train=False, num_actions=27, train_backbone=False)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_batch, shuffle=False
    ) 
    best_loss = 100.0
    best_AP = 0
    timer = Timer()
    bboxes_error_data = []
    person_num_error_data = []
    
    val_folder_path = cfg_file.split('.')[0] + Loss
    if not os.path.isdir(val_folder_path):
        os.mkdir(val_folder_path)
    
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',patience=2)
    def func(epoch):
            return 0.9**(epoch)


    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=func)
    for epochs in range(epoch):
        model.train()

        #############################################train#############################################
        batch_Loss = torch.zeros((1,), requires_grad=True).cuda()
        for ii, batch in enumerate(tqdm(dataloader)):

            (images, bboxes, pcs, bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id, action,social_group_activity, data_dict) = batch 
            batch_size = images.shape[0]
            person_num = get_num_person(person_id=person_id)
            for num in person_num:
                if num < 2:

                    person_num_error_data.append(ii)
                    
            if ii in person_num_error_data:
                continue

            res = model(batch)
            A_theta, pose_1, pose_2, pose_3, intrctn_1, intrctn_2, intrctn_3, intrctn_4,  \
                SG_pose_1, SG_pose_2, SG_pose_3, SG_intrctn_1, SG_intrctn_2, SG_intrctn_3, SG_intrctn_4,card = res
    
            A_theta_list = []
            pose_1_list = []
            pose_2_list = []
            pose_3_list = []
            intrctn_1_list = []
            intrctn_2_list = []
            intrctn_3_list = []
            intrctn_4_list = []
            card_list = []
            
            SG_pose_1_list = []
            SG_pose_2_list = []
            SG_pose_3_list = []
            SG_intrctn_1_list = []
            SG_intrctn_2_list = []
            SG_intrctn_3_list = []
            SG_intrctn_4_list = []

            for b in range(batch_size):
                
                A_theta_list.append(A_theta[b,:person_num[b],:person_num[b]])
                pose_1_list.append(pose_1[b,:person_num[b]])
                pose_2_list.append(pose_2[b,:person_num[b]])
                pose_3_list.append(pose_3[b,:person_num[b]])
                intrctn_1_list.append(intrctn_1[b,:person_num[b]])
                intrctn_2_list.append(intrctn_2[b,:person_num[b]])
                intrctn_3_list.append(intrctn_3[b,:person_num[b]])
                intrctn_4_list.append(intrctn_4[b,:person_num[b]])
                card_list.append(card[b])
                
                SG_pose_1_list.append(SG_pose_1[b,:person_num[b]])
                SG_pose_2_list.append(SG_pose_2[b,:person_num[b]])
                SG_pose_3_list.append(SG_pose_3[b,:person_num[b]])
                SG_intrctn_1_list.append(SG_intrctn_1[b,:person_num[b]])
                SG_intrctn_2_list.append(SG_intrctn_2[b,:person_num[b]])
                SG_intrctn_3_list.append(SG_intrctn_3[b,:person_num[b]])
                SG_intrctn_4_list.append(SG_intrctn_4[b,:person_num[b]])
            

            
            for A_th in A_theta_list:
                if np.isnan(torch.max(A_th).item()):
                    bboxes_error_data.append(ii)
                    continue
            if ii in bboxes_error_data:
                continue
            
            
            action = action.cuda()
            social_group_activity = social_group_activity.cuda()
            
            social_group_num = get_num_social_group(social_group_id=social_group_id)
            A_hat_list = get_adjacency(social_group_id,person_num)
            label = get_label_from_action(action, person_num)
            SG_label = get_label_from_action(social_group_activity, person_num)
            
           
            BCELoss = nn.BCELoss()
            BCELoss_NOT_reduce = nn.BCELoss(reduce=False)
            MSELoss = nn.MSELoss(reduction='mean')
            
            for i in range(batch_size):
                
                mask = torch.ones(A_theta_list[i].shape[0], A_theta_list[i].shape[0])
                mask[torch.eye(A_theta_list[i].shape[0]).bool()] = 0.
                mask = mask.to(A_theta_list[i].device)
                
                
                non_group_mask = A_hat_list[i] == 0
                
                Num_group_note = (A_hat_list[i] * mask).sum()
                Num_total_note = mask.sum()
                ratio = (Num_total_note - Num_group_note) / (3*Num_group_note + 1)
                
                
                L_bce2 = ratio * (BCELoss_NOT_reduce(A_theta_list[i], A_hat_list[i]) * mask) * A_hat_list[i] + (BCELoss_NOT_reduce(A_theta_list[i], A_hat_list[i]) * mask)*non_group_mask
                L_bce2 = L_bce2.sum() / (mask.sum())
                # L_bce = (BCELoss(A_theta_list[i], A_hat_list[i]) * mask).sum() / mask.sum()
                L_bce = BCELoss(A_theta_list[i], A_hat_list[i])
            
            
            BCELoss = nn.BCELoss()
            
            L_eig = get_eig_loss2(A_theta_list=A_theta_list,A_hat_list=A_hat_list)
            L_mse = MSELoss(torch.cat(card_list), torch.tensor(social_group_num).float().to(A_theta_list[0].device))
            L_g = L_bce + L_eig + L_mse

            ### 2.act loss
            CELoss = nn.CrossEntropyLoss()
            pose_loss_weight = [1., 1., 1.]         # lamda i
            interaction_loss_weight = [1., 1., 1., 1.]    # lamda j

            L_pose = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
            for i in range(batch_size):
                L_pose =  (CELoss(pose_1_list[i], label[0][i]) +  CELoss(pose_2_list[i], label[1][i]) + CELoss(pose_3_list[i], label[2][i]))
                        

                            

            L_interaction = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
            for i in range(batch_size):
                L_interaction += (BCELoss(intrctn_1_list[i], label[3][i]) + \
                                    BCELoss(intrctn_2_list[i], label[4][i]) + \
                                    BCELoss(intrctn_3_list[i], label[5][i]) + \
                                    BCELoss(intrctn_4_list[i], label[6][i]) )
                                    
            L_act = L_pose + L_interaction

            ### 3.social group activity loss
            CELoss = nn.CrossEntropyLoss()
            SG_L_pose = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
            for i in range(batch_size):
                SG_L_pose =  (BCELoss(SG_pose_1_list[i], SG_label[0][i]) +  BCELoss(SG_pose_2_list[i], SG_label[1][i]) + BCELoss(SG_pose_3_list[i], SG_label[2][i]))
                        

                            

            SG_L_interaction = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
            for i in range(batch_size):
                SG_L_interaction += (BCELoss(SG_intrctn_1_list[i], SG_label[3][i]) + \
                                    BCELoss(SG_intrctn_2_list[i], SG_label[4][i]) + \
                                    BCELoss(SG_intrctn_3_list[i], SG_label[5][i]) + \
                                    BCELoss(SG_intrctn_4_list[i], SG_label[6][i]) )
                                    
            SG_L_act = SG_L_pose + SG_L_interaction




            ### total loss
            if Loss == "L_g":
                L_total = L_g
            elif Loss == "L_bce":
                L_total = L_bce
                
            elif Loss == "L_bce2":
                L_total = L_bce2
                
            elif Loss == "L_total":
                L_total = L_bce + L_act + SG_L_act
            elif Loss == "L_act":
                L_total = L_act + SG_L_act
            
            # L_total =  L_pose + L_interaction + L_bce + L_eig + L_mse

            batch_Loss += L_total
            
            if ii % 8 == 0:
                if ii == 0: pass
                batch_Loss /= 8.
                run["train/batch_Loss"].append(batch_Loss.item())
                optimizer.zero_grad()
                batch_Loss.backward()
                optimizer.step()
                batch_Loss =torch.zeros((1,), requires_grad=True).cuda()


        
                
                run["train/L_bce"].append(L_bce.item())
                run["train/L_pose"].append(L_pose.item())
                run["train/L_interaction"].append(L_interaction.item())
                run["train/SG_L_act"].append(SG_L_act)
        #############################################val#############################################
        model.eval()
        val_loss = 0
        val_act_loss = 0
        val_activity_loss = 0
        res_txt_path = val_folder_path + f'/epoch{epochs}'
        (social_grouping_result_path, action_result_path, activity_result_path) = constrct_group(valloader, model, res_txt_path)
        
        
        

        with open(labelmap_path['task_3']) as l , open(gt_file_path["group"]) as g , open(social_grouping_result_path) as d:
            res = eval.evaluate(labelmap=l, groundtruth=g,detections=d,task="task_3")
        table = analysis_result(res, mode='task3')

        
        run["val/G1_AP"].append(table['G1_AP'])
        run["val/G2_AP"].append(table['G2_AP'])
        run["val/G3_AP"].append(table['G3_AP'])
        run["val/G4_AP"].append(table['G4_AP'])
        run["val/G5_AP"].append(table['G5_AP'])
        run["val/overall_AP"].append(table['overall_AP'])
        
        grouping_AP = table['overall_AP']

        with open(labelmap_path['task_1']) as l , open(gt_file_path["action"]) as g , open(action_result_path) as d:
            res = eval.evaluate(labelmap=l, groundtruth=g,detections=d,task="task_1")
        AP = analysis_result(res, mode='task1')
        run["val/action_AP"].append(AP)
        
        action_AP = AP
        
        with open(labelmap_path['task_4']) as l , open(gt_file_path["activity"]) as g , open(activity_result_path) as d:
            res = eval.evaluate(labelmap=l, groundtruth=g,detections=d,task="task_4")
        AP = analysis_result(res, mode='task4')
        run["val/activity_AP"].append(AP)

        activity_AP = AP

        if Loss == 'L_bce':
            score = grouping_AP
        else:
            score = activity_AP * grouping_AP
        
        
        if score > best_AP:
            best_AP = score
            best_AP_ckpt = {"model" : model.state_dict(), 
                             "epoch" : epochs,
                             "overall_AP" : grouping_AP,
                             "Loss" : Loss}
        torch.save(best_AP_ckpt, val_folder_path + "/best_AP_ckpt.pth")
        scheduler.step()
        with torch.no_grad():
            for ii, batch in enumerate(tqdm(valloader)):
                break
                
                (images, bboxes, pcs, bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id, action, social_group_activity,data_dict) = batch 
                batch_size = images.shape[0]
                person_num = get_num_person(person_id=person_id)
                for num in person_num:
                    if num < 2:
                        person_num_error_data.append(ii)
                        
                if ii in person_num_error_data:
                    continue
                    

                res = model(batch)
                A_theta, pose_1, pose_2, pose_3, intrctn_1, intrctn_2, intrctn_3, intrctn_4,  \
                SG_pose_1, SG_pose_2, SG_pose_3, SG_intrctn_1, SG_intrctn_2, SG_intrctn_3, SG_intrctn_4,card = res
        
                A_theta_list = []
                pose_1_list = []
                pose_2_list = []
                pose_3_list = []
                intrctn_1_list = []
                intrctn_2_list = []
                intrctn_3_list = []
                intrctn_4_list = []
                card_list = []
                    
                SG_pose_1_list = []
                SG_pose_2_list = []
                SG_pose_3_list = []
                SG_intrctn_1_list = []
                SG_intrctn_2_list = []
                SG_intrctn_3_list = []
                SG_intrctn_4_list = []

                for b in range(batch_size):
                    
                    A_theta_list.append(A_theta[b,:person_num[b],:person_num[b]])
                    pose_1_list.append(pose_1[b,:person_num[b]])
                    pose_2_list.append(pose_2[b,:person_num[b]])
                    pose_3_list.append(pose_3[b,:person_num[b]])
                    intrctn_1_list.append(intrctn_1[b,:person_num[b]])
                    intrctn_2_list.append(intrctn_2[b,:person_num[b]])
                    intrctn_3_list.append(intrctn_3[b,:person_num[b]])
                    intrctn_4_list.append(intrctn_4[b,:person_num[b]])
                    card_list.append(card[b])
                    
                    SG_pose_1_list.append(SG_pose_1[b,:person_num[b]])
                    SG_pose_2_list.append(SG_pose_2[b,:person_num[b]])
                    SG_pose_3_list.append(SG_pose_3[b,:person_num[b]])
                    SG_intrctn_1_list.append(SG_intrctn_1[b,:person_num[b]])
                    SG_intrctn_2_list.append(SG_intrctn_2[b,:person_num[b]])
                    SG_intrctn_3_list.append(SG_intrctn_3[b,:person_num[b]])
                    SG_intrctn_4_list.append(SG_intrctn_4[b,:person_num[b]])
                
                

                
                for A_th in A_theta_list:
                    if np.isnan(torch.max(A_th).item()):
                        bboxes_error_data.append(ii)
                        continue
                if ii in bboxes_error_data:
                    continue
                
                
                action = action.cuda()
                social_group_activity = social_group_activity.cuda()
                
                
                social_group_num = get_num_social_group(social_group_id=social_group_id)
                A_hat_list = get_adjacency(social_group_id,person_num)
                label = get_label_from_action(action, person_num)
                SG_label = get_label_from_action(social_group_activity, person_num)
            
                BCELoss = nn.BCELoss()
                MSELoss = nn.MSELoss(reduction='mean')
                L_bce = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
                for i in range(batch_size):
                    L_bce = BCELoss(A_theta_list[i],A_hat_list[i]) 
                
                val_loss += L_bce
                
                
                CELoss = nn.CrossEntropyLoss()

                L_pose = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
                for i in range(batch_size):
                    L_pose =  (CELoss(pose_1_list[i], label[0][i]) +  CELoss(pose_2_list[i], label[1][i]) + CELoss(pose_3_list[i], label[2][i]))

                L_interaction = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
                for i in range(batch_size):
                    L_interaction += (BCELoss(intrctn_1_list[i], label[3][i]) + \
                                        BCELoss(intrctn_2_list[i], label[4][i]) + \
                                        BCELoss(intrctn_3_list[i], label[5][i]) + \
                                        BCELoss(intrctn_4_list[i], label[6][i]) )
                                        
                val_act_loss += L_pose + L_interaction
                
                ### 3.social group activity loss
                CELoss = nn.CrossEntropyLoss()
                SG_L_pose = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
                for i in range(batch_size):
                    SG_L_pose =  (BCELoss(SG_pose_1_list[i], SG_label[0][i]) +  BCELoss(SG_pose_2_list[i], SG_label[1][i]) + BCELoss(SG_pose_3_list[i], SG_label[2][i]))
                            

                                

                SG_L_interaction = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
                for i in range(batch_size):
                    SG_L_interaction += (BCELoss(SG_intrctn_1_list[i], SG_label[3][i]) + \
                                        BCELoss(SG_intrctn_2_list[i], SG_label[4][i]) + \
                                        BCELoss(SG_intrctn_3_list[i], SG_label[5][i]) + \
                                        BCELoss(SG_intrctn_4_list[i], SG_label[6][i]) )
                                        
                                        
                
      
                
                val_activity_loss += SG_L_pose + SG_L_interaction
            
            # val_loss = val_loss.item() / len(valloader)
            # val_activity_loss = val_activity_loss.item() / len(valloader)
            # val_act_loss = val_act_loss.item() / len(valloader)
            # run["val/val_loss"].append(val_loss)
            # run["val/act_loss"].append(val_act_loss)
            # run["val/SG_act_loss"].append(val_activity_loss)
            
            # L_act = val_activity_loss
            
            
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     best_ckpt = {"model" : model.state_dict(), 
            #                  "epoch" : epochs,
            #                  "val_loss" : val_loss,
            #                  "act_loss" : L_act}

                
            
        
        
        torch.save(model.state_dict(), val_folder_path +'/best_val_loss.pth')
        run['parameters/poor_num_person_idx'] = person_num_error_data
        run['parameters/bboxes_error_idx'] =bboxes_error_data
        print(f"epoch : {epochs} done / epoch time : {timer.epochtime()}")

    return






print("cfg path : ", cfg_file)

run = neptune.init_run(
    project="GAR/mil",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwY2E0ZjJlZS1kMGYzLTQzMTItYjI5Yi1jZDcyNjg3Nzc0NmEifQ==",
)  # your credentials
params = {"cfg_file": cfg_file,
          "detail" : cfg.TRAINER.model_detail}
run["parameters"] = params

run["cfg"].upload(cfg_file)
run["gat_model"].upload('/mnt/server6_hard1/donguk/Multimodal_GAR/model/gat_model.py')

batch_size = cfg.TRAINER.BATCH_SIZE
# dataset, dataloader
dataset = JRDB_act(cfg.DATALOADER.train.augmentation,
                       root_path='/mnt/server14_hard1/jiny/GAR/Datasets/JRDB/',
                       is_train=True, num_actions=27, train_backbone=False)
dataloader = DataLoader(
    dataset, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=dataset.collate_batch, shuffle=cfg.TRAINER.IS_SHUFFLE
) 



model = GAR_Fusion_ALL(cfg, dataset)
model = nn.DataParallel(model, output_device=0)



# model load and freeze 
# model_path = "/mnt/server6_hard1/donguk/Multimodal_GAR/Multimodal_cfg/Attention_mat4L_total/best_AP_ckpt.pth"
# model.load_state_dict(torch.load(model_path)['model'])

# # for para in model.parameters():
#     para.requires_grad = False

# for name, para in model.module.GAR_model.named_parameters():
#     if name.split('.')[0] in ["D_embed", "card_net", "bn_rgb", "bn_lidar", "AttFusModule"]:
#         pass
#     else:
#         para.requires_grad = True





model.cuda()
# scheduler
model.train()






# for para in model.parameters():
#     para.requires_grad = False

# for name, para in model.module.GAR_model.named_parameters():
#     para.requires_grad = False
#     if name.split('.')[0] in ["AttFusModule", "phi", "sigma", "bn_rgb", "bn_lidar"]:
#         para.requires_grad = True


### STAGE_1 
optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINER.STAGE_1.LEARNING_RATE)
train_net(dataloader, model, optimizer, cfg.TRAINER.STAGE_1.EPOCH,cfg.TRAINER.STAGE_1.LOSS, run)

## STAGE_2
optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINER.STAGE_2.LEARNING_RATE)
train_net(dataloader, model, optimizer, cfg.TRAINER.STAGE_2.EPOCH,cfg.TRAINER.STAGE_2.LOSS, run)


torch.save(model.state_dict(), cfg.TRAINER.SAVE_PATH)
run.stop()

