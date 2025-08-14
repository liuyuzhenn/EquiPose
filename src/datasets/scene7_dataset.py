from torch.utils.data import Dataset
import torch

from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype(np.float32)
STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)).astype(np.float32)


class Scene7Dataset(Dataset):
    def __init__(self, configs, split):
        super(Scene7Dataset, self).__init__()
        self.configs = configs
        self.data_dir = configs['path']
        self.list_path = configs['list_path']
        self.split = split
        self.height = configs.get('height', 256)
        self.width = configs.get('width', 256)

        self.data = self.get_metas()

    def process_epoch(self):
        pass


    def get_metas(self):
        metas = []
        folder = os.path.join(self.list_path)
        
        f = np.load(os.path.join(folder, self.split+'.npz'), allow_pickle=True)
        name, score = f['name'], f['score']
        metas += [[n[0], n[1], n[2], n[3], s] for n, s in zip(name, score)]
        
        return metas

    def read_img(self, filename):
        img = Image.open(filename)
        w, h = img.width, img.height
        img = img.resize([self.width, self.height])
        img = np.array(img, dtype=np.float32)/255.
        img = (img-MEAN)/STD
        # scale 0~255 to 0~1
        np_img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return np_img, w, h

    def read_K(self, w, h):
        K = np.array([
            [585, 0, 320],
            [0, 585, 240],
            [0, 0, 1],
            ], dtype=np.float32)
        K[0, :] *= self.width/w
        K[1, :] *= self.height/h
        return K.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        scene, sub = self.data[index][:2]
        scene = '{}/seq-{:0>2}'.format(scene, sub)
        id1, id2, score = self.data[index][2:]
        
        frame1 = "frame-{:0>6}.".format(id1)
        frame2 = "frame-{:0>6}.".format(id2)
        img1_path = os.path.join(self.data_dir, scene, frame1+'color.png')
        img2_path = os.path.join(self.data_dir, scene, frame2+'color.png')
        pose1_path = os.path.join(self.data_dir, scene, frame1+'pose.txt')
        pose2_path = os.path.join(self.data_dir, scene, frame2+'pose.txt')

        img1, w_o, h_o = self.read_img(img1_path)
        img2, _, _ = self.read_img(img2_path)
        imgs = torch.stack([img1, img2], dim=0).float()

        pose1 = np.loadtxt(pose1_path).astype(np.float32)
        pose2 = np.loadtxt(pose2_path).astype(np.float32)
        K = self.read_K(w_o, h_o)

        rel_trans = np.linalg.inv(pose2)@pose1
        R = rel_trans[:3, :3]
        t = rel_trans[:3, 3]

        return {
            'images': imgs,
            'rotation': R,
            'translation': t,
            'score': score,
            'intrinsics': np.stack((K, K), axis=0)
        }
