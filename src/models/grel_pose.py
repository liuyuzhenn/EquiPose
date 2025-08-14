# from: Learning to Localize in Unseen Scenes with Relative Pose Regressors
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from .utils import *

from .loftr.loftr import LoFTR
from .loftr.config import _CN as loftr_cfg
from yacs.config import CfgNode as CN

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

class GrelPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.configs = config
        model_configs = config['model_configs']
        self.equivariance_mode = model_configs.get('equivariance_mode', False)
        self.rotation_mode = model_configs.get('rotation_mode', 'quaternion')
        loftr_model = LoFTR(lower_config(loftr_cfg)['loftr'])
        self.rotation_mode = model_configs.get('rotation_mode', 'quaternion')
        # LoFTR only for feature extraction
        self.backbone = loftr_model.backbone
        self.pos_encoding = loftr_model.pos_encoding
        self.loftr_coarse = loftr_model.loftr_coarse
        
        train_full = model_configs.get('train_full', False)
        self.backbone.requires_grad_(train_full)
        self.pos_encoding.requires_grad_(train_full)
        self.loftr_coarse.requires_grad_(train_full)
        
        # build ResNet block according to the supp.
        self.resblock = nn.Sequential(PreActBlock(517, 256, 2), 
                                    PreActBlock(256, 128, 2),
                                      nn.BatchNorm2d(128), 
                                      nn.ReLU(),
                                      nn.Conv2d(128, 16, kernel_size=1))

        if self.rotation_mode == 'quaternion':
            self.pose_size = 4+3
        elif self.rotation_mode == '6D':
            self.pose_size = 6+3
        else:
            raise ValueError('rotation model could only be quaternion | 6D!')
        self.out_head = nn.Sequential(
            nn.Linear(1408, 128),
            nn.ReLU(),
            nn.Linear(128, self.pose_size),
        )

        MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).float()
        STD = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).float()
        self.register_buffer('MEAN', MEAN)
        self.register_buffer('STD', STD)
        
        # if config['dataset_configs']['height']>config['dataset_configs']['width']:
        #     h, w = 42, 32
        # else:
        h, w = 32, 42
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij") 
        grids = torch.stack((xv, yv), 2) # 32, 42, 2
        self.register_buffer('grids', grids)
        self.register_buffer('grids_vec', grids.reshape(-1, 2))
    
    def feature_agg(self, feat1, feat2):
        B, D, H, W = feat1.shape
        vol0 = feat1.view(B, D, H * W)
        vol1 = feat2.view(B, D, H * W)

        vol0_n = torch.nn.functional.normalize(vol0, dim=1)
        vol1_n = torch.nn.functional.normalize(vol1, dim=1)
        corr_volume = torch.bmm(vol0_n.transpose(2, 1), vol1_n) # B, N, N
        corr_volume = torch.softmax(corr_volume, dim=-1)
        # inds_max = torch.argmax(corr_volume, dim=-1) # B, N
        val_max, inds_max = corr_volume.max(dim=-1) # B, N
        
        # vol1_warped = vol1.permute(0,2,1).reshape(-1, D)[inds_max.reshape(-1)].reshape(B,H*W,D).permute(0, 2, 1)
        vol1_warped = torch.stack([vol1[i, :, inds] for i,inds in enumerate(inds_max)], dim=0)
        
        pts_warped = torch.stack([self.grids_vec]*B, dim=0) # B,N,2
        pts_warped = torch.stack([pts_warped[i, inds] for i,inds in enumerate(inds_max)], dim=0)

        feat2_warped = vol1_warped.reshape(B, D, H, W)
        pts_warped_b2hw = pts_warped.permute(0, 2, 1).reshape(B, 2, H, W)
        pts_b2hw = torch.stack([self.grids]*B, dim=0).permute(0,3,1,2)
        
        val_max = val_max.reshape(B, 1, H, W)
        feat_agg = torch.concat([feat1, pts_b2hw, feat2_warped, pts_warped_b2hw, val_max], dim=1)
        
        vec_agg = self.resblock(feat_agg)
        pred = self.out_head(vec_agg.flatten(1))
        q, t = pred[:, :self.pose_size-3], pred[:, self.pose_size-3:]
        if self.rotation_mode=='quaternion':
            R = quaternion_to_matrix(q)
        elif self.rotation_mode == '6D':
            R = rotation_6d_to_matrix(q)

        return R, t
    
    def forward(self, inputs_data, mode='train'):
        images = inputs_data['images'] # B,2,3,H,W
        image0 = images[:, 0] # B, 1, H, W
        image1 = images[:, 1] # B, 1, H, W
        if image0.shape[2]<=image0.shape[3]:
            image0 = F.interpolate(image0, size=(256, 336))
            image1 = F.interpolate(image1, size=(256, 336))
        else:
            image0 = F.interpolate(image0, size=(336, 256))
            image1 = F.interpolate(image1, size=(336, 256))
        
        # convert to gray scale
        image0 = torch.mean(image0*self.STD+self.MEAN, dim=1, keepdim=True)
        image1 = torch.mean(image1*self.STD+self.MEAN, dim=1, keepdim=True)
        
        data = {'image0': image0, 
                 'image1': image1}
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        # 1. Local Feature CNN
        feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')


        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        feat_c0_, feat_c1_ = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        hf, wf = feats_c.shape[2:]
        feat_c0_bchw = rearrange(feat_c0_.reshape(-1, hf, wf, 256), 'n h w c -> n c h w')
        feat_c1_bchw = rearrange(feat_c1_.reshape(-1, hf, wf, 256), 'n h w c -> n c h w')

        R1, t1 = self.feature_agg(feat_c0_bchw, feat_c1_bchw)
        if self.equivariance_mode:
            feat_c1_inv, feat_c0_inv = self.loftr_coarse(feat_c1, feat_c0, mask_c1, mask_c0)
            feat_c0_inv_bchw = rearrange(feat_c0_inv.reshape(-1, hf, wf, 256), 'n h w c -> n c h w')
            feat_c1_inv_bchw = rearrange(feat_c1_inv.reshape(-1, hf, wf, 256), 'n h w c -> n c h w')

            R2, t2 = self.feature_agg(feat_c1_inv_bchw, feat_c0_inv_bchw)
            
            R2 = R2.transpose(-1, -2)
            R1tR2 = torch.bmm(R1.transpose(-1,-2), R2)
            R1tR2_sqr = axis_angle_to_matrix(matrix_to_axis_angle(R1tR2)/2)
            R = torch.bmm(R1, R1tR2_sqr)

            t = (t1-torch.bmm(R, t2.unsqueeze(-1)).squeeze(-1))/2
            
            return {'rotation': R, 'translation': t}
        
        return {'rotation': R1, 'translation': t1}