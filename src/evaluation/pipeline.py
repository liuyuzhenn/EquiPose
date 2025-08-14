from math import pi
import os
import numpy as np
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from importlib import import_module
import matplotlib.pyplot as plt
import torch
from src.evaluation.utils import error_auc, result_pretty_format
from src.models.utils import *
from src.utils import to_device
from src import load_configs
import sys
sys.path.append('.')


MEAN = np.array([0.485, 0.456, 0.406]).reshape(
    (3, 1, 1)).astype(np.float32)
STD = np.array([0.229, 0.224, 0.225]).reshape(
    (3, 1, 1)).astype(np.float32)


def _name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))


class Evaluation:
    def __init__(self, dataset_configs, configs):
        self.configs = configs
        self.workspace = configs['workspace']
        self.thresholds = configs.get('thresholds', '3,5,10')
        self.thresholds = [int(t) for t in self.thresholds.split(',')]
        self.thresholds_t = configs.get('thresholds', '0.1,0.2,0.5')
        self.thresholds_t = [float(t) for t in self.thresholds_t.split(',')]
        if self.workspace != '':
            os.makedirs(self.workspace, exist_ok=True)
        self.dataset_configs = dataset_configs
        dataset_cls = import_module('.datasets.{}'.format(
            dataset_configs['name']), 'src')
        dataset_cls = getattr(dataset_cls, _name_to_class(
            dataset_configs['name']))
        dataset = dataset_cls(dataset_configs, configs.get('split', 'test'))
        self.data_loader = DataLoader(
            dataset, batch_size=dataset_configs['batch_size'], shuffle=False)

    def get_pose(self, data):
        raise NotImplementedError

    def get_metrics(self, errors_R, errors_t):
        mn = np.mean(errors_R)
        mn_t = np.mean(errors_t)

        metrics = {
            'R_mean': float(mn),
            't_mean': float(mn_t),
        }
        R_metrics = error_auc(errors_R, self.thresholds)
        R_metrics = {'R_'+k: float(v) for k, v in R_metrics.items()}
        t_metrics = error_auc(errors_t, self.thresholds_t)
        t_metrics = {'t_'+k: float(v) for k, v in t_metrics.items()}

        metrics.update(R_metrics)
        metrics.update(t_metrics)
        return metrics

    def evaluate(self) -> None:
        errors_R = []
        errors_t = []
        Rs = []
        ts = []
        Rgts = []
        tgts = []
        bar = tqdm(self.data_loader)
        for data in bar:
            imgs: np.ndarray = data['images'].numpy()  # 1,2,3,H,W
            assert imgs.shape[0] == 1, "Only accept inputs with batch_size=1"

            # compute pose
            R, t = self.get_pose(data)

            # evalutae
            R_gt = data['rotation'][0].numpy()  # 3,3
            t_gt = data['translation'][0]  # 3
            Rgts.append(R_gt)
            tgts.append(t_gt.numpy())
            Rs.append(R)
            ts.append(t)
            t_gt = t_gt.numpy()
            t_gt = -R_gt.T@t_gt.reshape((3,1))
            t = -R.T@t.reshape((3,1))

            angle = np.clip((np.trace(R@R_gt.T)-1)/2, -1, 1)
            angle = np.arccos(angle)
            angle = float(angle/np.pi*180)
            t_error = np.linalg.norm(t-t_gt, ord=2)
            bar.set_description(
                'rotation: {:.2f}, translation: {:.2f}'.format(angle, t_error))
            errors_R.append(angle)
            errors_t.append(t_error)

        errors_R = np.array(errors_R)
        errors_t = np.array(errors_t)
        Rs = np.array(Rs)
        ts = np.array(ts)
        Rgts = np.array(Rgts)
        tgts = np.array(tgts)
        metrics = self.get_metrics(errors_R, errors_t)

        lines = result_pretty_format({"Default": metrics})
        for l in lines:
            print(l)

        if self.workspace == '':
            return

        with open(os.path.join(self.workspace, 'dataset_configs.yml'), 'w') as f:
            yaml.dump(self.dataset_configs, f, default_style=False)
        with open(os.path.join(self.workspace, 'configs.yml'), 'w') as f:
            yaml.dump(self.configs, f, default_style=False)
        with open(os.path.join(self.workspace, 'result.yml'), 'w') as f:
            yaml.dump(metrics, f, default_style=False)

        error_path = os.path.join(self.workspace, "data.npz")
        np.savez(error_path, errors_R=errors_R, errors_t=errors_t, Rs=Rs, ts=ts, Rgts=Rgts, tgts=tgts)


class EvaluationLearningBased(Evaluation):
    def __init__(self, dataset_configs, configs):
        super().__init__(dataset_configs, configs)
        self.cfg_path = configs['cfg_path']
        self.ckpt_path = configs['ckpt_path']
        self.configs = load_configs(self.cfg_path)
        self.model_configs = self.configs['model_configs']
        self.model = self.get_model()
        self.model.eval()
        self.equivariance_inference = configs.get(
            'equivariance_inference', False)
        print('Equivariance inference: {}'.format(self.equivariance_inference))

    def get_model(self):
        project = self.configs.get('project', 'src')
        model_module = import_module('.models.{}'.format(
            self.model_configs['name']), project)
        model = getattr(model_module, _name_to_class(
            self.model_configs['name']))(self.configs)
        mdict = torch.load(self.ckpt_path, map_location='cpu')
        model.load_state_dict(mdict['model_state_dict'])
        model = model.to('cuda')
        return model

    @torch.no_grad()
    def get_pose(self, data):
        data = to_device(data, 'cuda')
        model_outputs = self.model(data)

        R: torch.Tensor = model_outputs['rotation'][0]
        t: torch.Tensor = model_outputs['translation'][0]

        if self.equivariance_inference:
            model_outputs = self.model(data)
            images_flipped = torch.flip(data['images'], [1])
            data['images'] = images_flipped
            model_outputs = self.model(data)
            R2 = model_outputs['rotation'][0].transpose(-1, -2)
            t2 = model_outputs['translation'][0]
            R1tR2 = R.T@R2
            R1tR2_sqr = axis_angle_to_matrix(matrix_to_axis_angle(R1tR2)/2)
            R = R@R1tR2_sqr
            t = (t-torch.mm(R, t2.unsqueeze(-1)).squeeze(-1))/2
        R = R.cpu().numpy()
        t = t.cpu().numpy()
        return R, t