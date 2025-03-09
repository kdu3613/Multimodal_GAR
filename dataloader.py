import os
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
from data.utils.utils import *
import data.utils.jrdb_transforms as jt
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
from collections import defaultdict
from torch.utils.data import DataLoader

class JRDB_act(data.Dataset):
    def __init__(self, config, root_path, is_train, num_actions, train_backbone):
        phase = 'train' if is_train else 'test'
        self.anns = np.load(root_path + 'train_dataset_with_activity/labels_2019/{}_annotations.npy'.format(phase), allow_pickle=True).item()
        self.frames = self._all_frames(self.anns)   #(seq_id, frame_id)

        self.image_path = os.path.join(root_path, 'train_dataset_with_activity/images/image_stitched')
        self.pc_path = os.path.join(root_path, 'train_dataset_with_activity/pointclouds/lower_velodyne')
        
        self.image_size = config.image_size
        # self.is_training = is_train
        self.is_training = True
        self.is_finetune = train_backbone

        self.num_actions = num_actions

        self.num_boxes = config.num_boxes
        self.num_frames = config.sample.num_frames
        self.feature_size = (112,12)
        
        self._num_points = config.point_cloud.num_points
        
        vs = config.point_cloud.voxel_size
        voxel_size = (
            np.array(vs, dtype=np.float32)
            if isinstance(vs, list)
            else np.array([vs, vs, vs], dtype=np.float32)
        )
        self._voxel_size = voxel_size.reshape(3, 1)
        self.class_names = ['Pedestrian']
        
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.transforms = transforms.Compose([transforms.Resize(self.image_size),
        #                                         transforms.ToTensor()])
        
        self.point_cloud_range = np.array(config.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            config.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            config.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.is_training, num_point_features=self.point_feature_encoder.num_point_features
        )
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        
        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
            
        self.vfe = MeanVFE(
            config,
            num_point_features=self.point_feature_encoder.num_point_features,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            depth_downsample_factor=self.depth_downsample_factor
        )
        
    def __getitem__(self, index):
        select_frames = self.get_frames(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        return sample

    def __len__(self):
        return len(self.frames)

    def _all_frames(self, anns):
        return [(s, f) for s in anns for f in anns[s]]

    def get_frames(self, frame):

        sid, src_fid = frame

        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid, src_fid + self.num_frames - 1)
                return [(sid, src_fid, fid)]

            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid, src_fid + self.num_frames)]

        else:
            if self.is_training:
                sample_frames = list(range(src_fid - self.num_frames//2, src_fid + self.num_frames//2 +1))
                # sample_frames = random.sample(range(src_fid - self.num_frames, src_fid + self.num_frames + 1), 1)
                return [(sid, src_fid, fid) for fid in sample_frames]
            else:
                sample_frames = list(range(src_fid - self.num_frames//2, src_fid + self.num_frames//2 +1))
                # sample_frames = random.sample(range(src_fid - self.num_frames, src_fid + self.num_frames + 1), 1)
                return [(sid, src_fid, fid) for fid in sample_frames]

    def one_hot(self, labels, num_categories):
        result = [0 for _ in range(num_categories)]
        for label in labels:
            result[label] = 1
        return result

    def load_pc(self, urls):
        
        pc_lower = load_pointcloud(urls)
        pc_upper = load_pointcloud(urls.replace('lower_velodyne','upper_velodyne'))
        
        # Translate the position of points 
        pc_upper[:3] = jt.transform_pts_upper_velodyne_to_base(pc_upper[:3])
        pc_lower[:3] = jt.transform_pts_lower_velodyne_to_base(pc_lower[:3])

        pc = np.concatenate([pc_upper, pc_lower], axis=0) 
        pc = get_lidar_with_sweeps(pc,self._num_points)        
        
        return pc
    
    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        

        OH, OW = self.feature_size

        images, bboxes, bboxes3d, pcs = [], [], [],[]
        #pc_offsets = []
        actions = []
        social_group_activity = []
        person_id = []
        social_group_id = []
        bboxes_num = []
        bboxes3d_num = []
        seq_id = []
        frame_id = []
        
        data_dicts = []
        # one_hot = torch.nn.functional.one_hot(unique_actions, cfg.num_actions).float()
        zero_action = [0 for _ in range(self.num_actions)]
        
        seq_names = sorted(os.listdir(self.image_path))
        
        
        ### if there is not file,,, change select frame as 0
        (sid, src_fid, fid) = select_frames[0]
        if not os.path.exists(self.image_path + '/' + seq_names[sid] + '/' + str(src_fid).zfill(6) + ".jpg"):
            select_frames = self.get_frames(self.frames[0])

        
        
       
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            this_image_path = self.image_path + '/' + seq_names[sid] + '/' + str(fid).zfill(6) + ".jpg"
            if os.path.exists(this_image_path):
                img = Image.open(this_image_path)
            else:
                img = Image.open(self.image_path + '/' + seq_names[int(sid)] + '/' + str(src_fid).zfill(6) + ".jpg")
            img = self.transforms(img)
            images.append(img)
            
            this_pc_path = os.path.join(self.pc_path,seq_names[sid],str(src_fid).zfill(6)+'.pcd')
            pc_ = self.load_pc(this_pc_path)
            pc_ = torch.from_numpy(pc_)           # openPCDet 에서 dict안에 points 는 np
            
            ###여기세 gt_boxes (3d)
            
            
            temp_boxes3d = []
            for box3d in self.anns[sid][src_fid]['bboxes_3d']:
                # [TODO] filter out corrupted annotations with negative dimension
                temp_boxes3d.append((box3d['cx'],box3d['cy'],box3d['cz'],box3d['l'],box3d['w'],box3d['h'],box3d['rot_z']))
            
            
            data_dict = {
                'points': pc_,
                'gt_boxes' : np.array(temp_boxes3d, dtype=np.float32)}
            
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(data_dict = data_dict)
            # data_dict = self.vfe.forward(data_dict)  
            
            
            
            
            # pcs.append(data_dict['voxel_features'])
            # pcs.append(data_dict['voxels'])
            data_dicts.append(data_dict)
            # print(data_dict["voxels"].shape)

            # if i > 0:
            #     continue
            temp_boxes = []
            ### 이거 이상하다!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for box in self.anns[sid][src_fid]['bboxes_2d']:
                # y1, x1, y2, x2 = box
                # w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                # temp_boxes.append((w1, h1, w2, h2))
                
                # x, y, w, h = box
                # x1, y1, x2, y2 = x - w/2, y - h/2 , x + w/2, y +h/2
                # x1, y1, x2, y2 = x1*self.image_size[1], y1*self.image_size[0], x2*self.image_size[1], y2*self.image_size[0]
                # temp_boxes.append((x1,y1,x2,y2))
                x, y, w, h = box
                x1, x2 = x, x + w
                y1, y2 = y, y + h
                x1, y1, x2, y2 = x1*self.image_size[1], y1*self.image_size[0], x2*self.image_size[1], y2*self.image_size[0]
                temp_boxes.append((x1,y1,x2,y2))

            temp_actions = self.anns[sid][src_fid]['actions'][:]
            temp_person_id = self.anns[sid][src_fid]['person_id'][:]
            temp_social_group_id = self.anns[sid][src_fid]['social_group_id'][:]
            
            temp_social_group_activity = self.anns[sid][src_fid]['social_group_activity'][:]
            
            
            bboxes_num.append(len(temp_boxes))
            bboxes3d_num.append(len(temp_boxes3d))
            
            temp_sid = []
            temp_fid = []
            for kk in range(len(temp_person_id)):
                temp_sid.append(sid)
                temp_fid.append(fid)

            # each batch have a same number of bounding boxes
            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_boxes3d.append((0, 0, 0, 0, 0, 0, 0))
                temp_actions.append(zero_action)
                temp_social_group_activity.append(zero_action)
                temp_person_id.append(-1)
                temp_social_group_id.append(-1)
                temp_fid.append(-1)
                temp_sid.append(-1)

            bboxes.append(temp_boxes)
            bboxes3d.append(temp_boxes3d)
            actions.append(temp_actions)
            social_group_activity.append(temp_social_group_activity)
            person_id.append(temp_person_id)
            social_group_id.append(temp_social_group_id)
            seq_id.append(temp_sid)
            frame_id.append(temp_fid)

        actions = np.array(actions[0], dtype=np.float32)
        social_group_activity = np.array(social_group_activity[0], dtype=np.float32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float32).reshape((-1, self.num_boxes, 4))
        bboxes3d = np.array(bboxes3d, dtype=np.float32).reshape((-1, self.num_boxes, 7))
        person_id = np.array(person_id, dtype=np.int32).reshape(-1, self.num_boxes)
        seq_id = np.array(seq_id, dtype=np.int32).reshape(-1, self.num_boxes)
        frame_id = np.array(frame_id, dtype=np.int32).reshape(-1, self.num_boxes)
        social_group_id = np.array(social_group_id, dtype=np.int32).reshape(-1, self.num_boxes)

        # convert to pytorch tensor
        images = torch.stack(images).float()
        # pcs = torch.stack(pcs).float()
        # pcs = np.array(pcs, dtype=np.float32)
        # pcs = torch.from_numpy(pcs).float()
        
        
        # pcs = torch.from_numpy(pcs[-1]).float()                     #got only last lidar data
        bboxes = torch.from_numpy(bboxes).float()
        bboxes3d = torch.from_numpy(bboxes3d).float()
        person_id = torch.from_numpy(person_id).long()
        seq_id = torch.from_numpy(seq_id).long()
        frame_id = torch.from_numpy(frame_id).long()
        social_group_id = torch.from_numpy(social_group_id).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        actions = torch.from_numpy(actions).float()
        social_group_activity = torch.from_numpy(social_group_activity).float()
        ### return value에서 -1의 의미는 15장의 프레임 중에서 마지막 15번째 프레임을 key frame으로 잡았다는 뜻입니다....
        
        return images, bboxes[-1], src_fid, bboxes3d[-1,:,:], bboxes_num, person_id[-1], social_group_id[-1], seq_id, frame_id,actions, social_group_activity,  data_dicts[-1]
    
    @staticmethod
    def collate_batch(batch_list, _unused=False):
       
    
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            
            data_dicts = cur_sample[-1]
            cur_sample = data_dicts
            
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        
        ####
        rgb_image = torch.stack([cur_sample[0] for cur_sample in batch_list]).float()
        bboxes = torch.stack([cur_sample[1] for cur_sample in batch_list]).float()
        src_fid = [cur_sample[2] for cur_sample in batch_list]
        bboxes3d = torch.stack([cur_sample[3] for cur_sample in batch_list]).float()
        bboxes_num = torch.stack([cur_sample[4] for cur_sample in batch_list]).float()
        person_id = torch.stack([cur_sample[5] for cur_sample in batch_list]).float()
        social_group_id = torch.stack([cur_sample[6] for cur_sample in batch_list]).float()
        seq_id = torch.stack([cur_sample[7] for cur_sample in batch_list]).float()
        frame_id = torch.stack([cur_sample[8] for cur_sample in batch_list]).float()
        actions = torch.stack([cur_sample[9] for cur_sample in batch_list]).float()
        social_group_activity = torch.stack([cur_sample[10] for cur_sample in batch_list]).float()
        ####

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'bm_points']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps", "overlap_mask", "depth_mask"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                            pad_value = 0
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)
                        elif key == "overlap_mask":                            
                            pad_width = (pad_h, pad_w)
                            pad_value = 0
                        elif key == "depth_mask":
                            pad_width = (pad_h, pad_w, (0, 0))
                            pad_value = 0

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                            pad_width=pad_width,
                                            mode='constant',
                                            constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ["point2img"]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['image_grid_dict', 'image_depths_dict', 'voxel_grid_dict', 'lidar_grid_dict']:
                    results = {}
                    for _val in val:
                        for _layer in _val:
                            if _layer in results:
                                results[_layer] = results[_layer].append(_val[_layer])
                            else:
                                results[_layer] = [_val[_layer]]
                    for _layer in results:
                        results[_layer] = torch.cat(results[_layer], dim=0)
                    ret[key] = results
                elif key in ['gt_dense']:
                    pass
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return rgb_image, bboxes,src_fid , bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id ,actions ,social_group_activity,ret

def get_num_person(person_id):
    _b = person_id.shape[0]
    res = []
    for i in range(_b):
        res.append(len(torch.unique(person_id[i])) - 1)
    return res

def get_adjacency(social_group_id, person_num):
    _b, _ = social_group_id.shape
    res = []
    for b in range(_b):
        social_group_id_b = social_group_id[b][:person_num[b]]
        res_b = torch.zeros((person_num[b], person_num[b]))
        for i in range(person_num[b]):
            for j in range(person_num[b]):
                if i == j:
                    res_b[i][j] = 1
                elif social_group_id_b[i] == social_group_id_b[j]:
                    res_b[i][j] = 1
                    res_b[j][i] = 1
        res.append(res_b.cuda())
    return res

record = torch.tensor([0,1]).unsqueeze(0)

from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    config = yaml.load(open('/mnt/server6_hard1/donguk/Multimodal_GAR/config/baseline.yaml'), Loader=yaml.FullLoader)
    config = EasyDict(config)

    dataset = JRDB_act(config.train.augmentation,
                       root_path='/mnt/server14_hard1/jiny/GAR/Datasets/JRDB/',
                       is_train=False, num_actions=27, train_backbone=False)
    dataloader = DataLoader(
    dataset, batch_size =1, pin_memory=True, num_workers=4, collate_fn=dataset.collate_batch, shuffle=True
    ) 
    
    for ii, sample in enumerate(dataloader):
        rgb_image, bboxes,src_fid , bboxes3d, bboxes_num, person_id, social_group_id, seq_id, frame_id ,actions ,social_group_activity,ret = sample
        person_num = get_num_person(person_id)
        
        bboxes3d = bboxes3d[0,:person_num[0],:2]
        A_hat = get_adjacency(social_group_id,person_num)[0]
        De = pairwise_euclidean_distance(bboxes3d, zero_diagonal=True)
        print("test")
        De = De.reshape(-1).unsqueeze(1)
        A_hat = A_hat.reshape(-1).unsqueeze(1)
        A_hat = A_hat.to("cpu")
        tmp = torch.concat([De,A_hat], dim=1)
        record = torch.concat([record,tmp], dim=0)
        
        
        if ii > 300:
            break
    
    x = record[:,0].numpy()
    y = record[:,1].numpy()
    
    scatter = pyp.scatter(x,y)
    pyp.show()
    
    
  
    
    