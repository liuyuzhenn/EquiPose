import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .utils import *
from .vit_modules.extractor import ResidualBlock
from .vit_modules.vision_transformer import _create_vision_transformer

class EightVit(nn.Module):
    def __init__(self, configs):
        super(EightVit, self).__init__()
        self.configs = configs
        model_configs = configs['model_configs']
        self.equivariance_mode = model_configs.get('equivariance_mode', False)
        # print('Equivariance mode: {}'.format(self.equivariance_mode))

        # hyperparams
        self.noess = None
        self.total_num_features = 192
        self.feature_resolution = (24, 24)
        # self.pose_size = 7
        self.rotation_mode = model_configs.get('rotation_mode', 'quaternion')
        if self.rotation_mode == 'quaternion':
            self.pose_size = 4 + 3
        elif self.rotation_mode == '6D':
            self.pose_size = 6 + 3
        self.num_patches = self.feature_resolution[0] * self.feature_resolution[1]
        extractor_final_conv_kernel_size = max(1, 28-self.feature_resolution[0]+1)
        self.pool_feat1 = min(96, 4 * 60)
        self.pool_feat2 = 60
        self.H2 = 512

        # layers
        self.flatten = nn.Flatten(0,1)
        self.resnet = models.resnet18(pretrained=True) # this will be overridden if we are loading pretrained model
        self.resnet.fc = nn.Identity()
        self.extractor_final_conv = ResidualBlock(128, self.total_num_features, 'batch', kernel_size=extractor_final_conv_kernel_size)

        self.num_heads = 3
        model_kwargs = dict(patch_size=16, embed_dim=self.total_num_features, depth=6, 
                            num_heads=self.num_heads, 
                            cross_features=False,
                            use_single_softmax=False,
                            no_pos_encoding=False,
                            noess=False, l1_pos_encoding=False)
        self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

        self.transformer_depth = 6
        self.fusion_transformer.blocks = self.fusion_transformer.blocks[:self.transformer_depth]
        self.fusion_transformer.patch_embed = nn.Identity()
        self.fusion_transformer.head = nn.Identity() 
        self.fusion_transformer.cls_token = None
        self.pos_encoding = None

        # we overwrite pos_embedding as we don't have class token
        self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,self.num_patches,self.total_num_features])) 
        # randomly initialize as usual 
        nn.init.xavier_uniform_(self.fusion_transformer.pos_embed) 

        pos_enc = 6
        self.H = int(self.num_heads*2*(self.total_num_features//self.num_heads + pos_enc) * (self.total_num_features//self.num_heads))
        self.pose_regressor = nn.Sequential(
            nn.Linear(self.H, self.H2), 
            nn.ReLU(), 
            nn.Linear(self.H2, self.H2), 
            nn.ReLU(), 
            nn.Linear(self.H2, self.pose_size),
        )


    def update_intrinsics(self, input_shape, intrinsics):
        sizey, sizex = self.feature_resolution
        scalex = sizex / input_shape[-1]
        scaley = sizey / input_shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,:,xidx] = scalex * intrinsics[:,:,xidx]
        intrinsics[:,:,yidx] = scaley * intrinsics[:,:,yidx]
            
        return intrinsics

    def extract_features(self, images, intrinsics=None):
        """ run feeature extraction networks """
        if intrinsics is not None:
            intrinsics = self.update_intrinsics(images.shape, intrinsics)

        # for resnet, we need 224x224 images
        input_images = self.flatten(images)
        input_images = F.interpolate(input_images, size=224)

        x = self.resnet.conv1(input_images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) 
        x = self.resnet.layer1(x) # 64, 56, 56
        x = self.resnet.layer2(x) # 128, 28, 28       
        
        x = self.extractor_final_conv(x) # 192, 24, 24 

        x = x.reshape([input_images.shape[0], -1, self.num_patches])
        features = x[:,:self.total_num_features]
        features = features.permute([0,2,1])

        return features, intrinsics
    

    def forward(self, inputs_data, mode='train'):
        """ Estimates SE3 between pair of frames """
        images = inputs_data['images'] # B,2,3,H,W
        B, _, _, _, _ = images.shape
        K = inputs_data.get('intrinsics', None)
        if K is not None:
            intrinsics = torch.zeros(B, K.shape[1], 4, device=K.device, dtype=K.dtype)
            intrinsics[:, :, 0] = K[:, :, 0, 0]
            intrinsics[:, :, 1] = K[:, :, 1, 1]
            intrinsics[:, :, 2] = K[:, :, 0, 2]
            intrinsics[:, :, 3] = K[:, :, 1, 2]
        else:
            intrinsics = None

        features, intrinsics = self.extract_features(images, intrinsics)
        _, d1, d2 = features.shape

        x = features[:,:,:self.total_num_features]
        x = self.fusion_transformer.patch_embed(x)
        x = x + self.fusion_transformer.pos_embed
        x = self.fusion_transformer.pos_drop(x)

        for layer in range(self.transformer_depth):
            x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)

        x = self.fusion_transformer.norm(x)

        pose_preds = self.pose_regressor(x.reshape([B, -1]))
        q1, t1 = pose_preds[:, :self.pose_size-3], pose_preds[:, self.pose_size-3:]

        # R1 = quaternion_to_matrix(q1)
        if self.rotation_mode == 'quaternion':
            R1 = quaternion_to_matrix(q1)
        else:
            R1 = rotation_6d_to_matrix(q1)

        if self.equivariance_mode:
            features2 = features.reshape((B, 2, d1, d2))
            features2 = self.flatten(torch.stack((features2[:,1], features2[:,0]), dim=1))

            # fusion
            x = features2[:,:,:self.total_num_features]
            x = self.fusion_transformer.patch_embed(x)
            x = x + self.fusion_transformer.pos_embed
            x = self.fusion_transformer.pos_drop(x)

            for layer in range(self.transformer_depth):
                x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)

            x = self.fusion_transformer.norm(x)

            pose_preds2 = self.pose_regressor(x.reshape([B, -1]))

            q2, t2 = pose_preds2[:, :self.pose_size-3], pose_preds2[:, self.pose_size-3:]
            if self.rotation_mode == 'quaternion':
                R2 = quaternion_to_matrix(q2)
            else:
                R2 = rotation_6d_to_matrix(q2)

            if self.normalize_T:
                t2 = F.normalize(t2, p=2, dim=-1)
            R2 = R2.transpose(-1, -2)

            R1tR2 = torch.bmm(R1.transpose(-1,-2), R2)
            R1tR2_sqr = axis_angle_to_matrix(matrix_to_axis_angle(R1tR2)/2)
            R = torch.bmm(R1, R1tR2_sqr)
            t = (t1-torch.bmm(R, t2.unsqueeze(-1)).squeeze(-1))/2
            return {'rotation': R, 'translation': t}
        else:
            return {'rotation': R1, 'translation': t1}
        
