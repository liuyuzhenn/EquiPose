from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).float()
STD = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).float()


class ScannetDataset(Dataset):
    def __init__(self, configs, split):
        super(ScannetDataset, self).__init__()
        self.configs = configs
        self.path = configs['path']
        self.split = split
        self.height = configs.get('height', 256)
        self.width = configs.get('width', 256)
        self.resize = transforms.Resize((self.height, self.width))

        # metas: scene_id, img1_id, img2_id
        self.list_path = configs['list_path']
        self.data = np.load(os.path.join(self.path, 'preprocessed', 'testdata', 'test.npz'))['name']
        self.data_dir = os.path.join(self.path, 'scans_test')

    def read_img(self, filename1, filename2):
        img1 = Image.open(filename1) # H, W, C [0,1]
        img2 = Image.open(filename2) # H, W, C [0,1]
        w, h = img1.width, img1.height
        img1 = torch.from_numpy(np.array(img1)/255.).float()
        img2 = torch.from_numpy(np.array(img2)/255.).float()
        imgs = torch.stack([img1, img2], dim=0).permute([0, 3, 1, 2]) # 2, C, H, W
        if self.split=='train':
            imgs = imgs.permute([1,2,3,0]) #  C, H, W, 2
            imgs = imgs.reshape([3, h, w*2])
            imgs = self.augcolor(imgs)
            imgs = imgs.reshape([3,h,w,2]).permute([3,0,1,2]).contiguous() # 2, C, H, W
        imgs = self.resize(imgs)
        imgs = (imgs-MEAN)/STD
        return imgs, w, h

    def read_K(self, filename, w, h):
        K = np.loadtxt(filename)[:3, :3]
        K[0, :] *= self.width/w
        K[1, :] *= self.height/h
        return K.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scene, sub, id1, id2,= self.data[index].astype('uint16')

        scene = 'scene{:0>4}_{:0>2}'.format(scene, sub)

        img1_path = os.path.join(self.data_dir, scene, 'color', f'{id1}.jpg')
        img2_path = os.path.join(self.data_dir, scene, 'color', f'{id2}.jpg')
        pose1_path = os.path.join(self.data_dir, scene, 'pose', f'{id1}.txt')
        pose2_path = os.path.join(self.data_dir, scene, 'pose', f'{id2}.txt')
        K_path = os.path.join(self.data_dir, scene,
                              'intrinsic', 'intrinsic_color.txt')

        imgs, w_o, h_o = self.read_img(img1_path, img2_path)

        pose1 = np.loadtxt(pose1_path).astype(np.float32)
        pose2 = np.loadtxt(pose2_path).astype(np.float32)
        K = self.read_K(K_path, w_o, h_o)

        rel_trans = np.linalg.inv(pose2)@pose1
        R = rel_trans[:3, :3]
        t = rel_trans[:3, 3]

        return {
            'images': imgs,
            'rotation': R,
            'translation': t,
            'intrinsics': np.stack((K, K), axis=0),
            'scene': scene,
            'id': np.array([id1, id2], dtype=np.int32),
        }