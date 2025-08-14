from math import pi
import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import *

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) if bn else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale,
                                      mode='bilinear', align_corners=True)
        return self.conv1(x)


class ResUNet(nn.Module):
    def __init__(self, cfgmodel, num_in_layers=3):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        self.in_planes = 64
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(
                num_in_layers, 64, kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False)
        else:
            self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # H/2
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4
        # encoder
        block_type = [PreActBlock, PreActBottleneck]
        block = block_type[cfgmodel['block']]
        num_blocks = [int(x) for x in cfgmodel['num_blocks'].strip().split("-")]
        self.encoder1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # H/4
        self.encoder2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # H/8
        self.encoder3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # H/16

        # decoder
        self.not_concat = getattr(cfgmodel, "not_concat", False)
        self.upconv4 = upconv(filters[2], 512, 3, 2)
        if not self.not_concat:
            self.iconv4 = conv(filters[1] + 512, 512, 3, 1)
        else:
            self.iconv4 = conv(512, 512, 3, 1)

        self.upconv3 = upconv(512, 256, 3, 2)
        if not self.not_concat:
            self.iconv3 = conv(filters[0] + 256, 256, 3, 1)
        else:
            self.iconv3 = conv(256, 256, 3, 1)

        num_out_layers = getattr(cfgmodel, "num_out_layers", 128)
        self.num_out_layers = num_out_layers
        self.outconv = conv(256, num_out_layers, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # encoding
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # decoding
        x = self.upconv4(x4)
        if not self.not_concat:
            x = self.skipconnect(x3, x)
        x = self.iconv4(x)

        x = self.upconv3(x)
        if not self.not_concat:
            x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.outconv(x)
        return x


class CorrelationVolumeWarping(torch.nn.Module):
    def __init__(self, volume_channels):
        super().__init__()

        self.position_encoder =  True # cfg.POSITION_ENCODER
        self.position_encoder_im1 = None # cfg.POSITION_ENCODER_IM1
        self.max_score_channel = True # cfg.MAX_SCORE_CHANNEL
        self.cv_out_layers = 0 # cfg.CV_OUTLAYERS
        self.cv_half_channels = False # cfg.CV_HALF_CHANNELS
        self.pos_encoder_channels = 0 # cfg.UPSAMPLE_POS_ENC
        self.dustbin = False # cfg.DUSTBIN
        self.normalise_dot_prod = False # cfg.NORMALISE_DOT
        
        self.num_out_layers = 2 * volume_channels
        self.num_out_layers += 2 if self.position_encoder else 0
        self.num_out_layers += 2 if self.position_encoder_im1 else 0
        self.num_out_layers += 1 if self.max_score_channel else 0

        if self.cv_out_layers > 0:
            self.CV_block = PreActBlock(4800, self.cv_out_layers)
            self.num_out_layers += self.cv_out_layers

        if self.pos_encoder_channels > 0:
            pos_encoder_input_channels = 0
            pos_encoder_input_channels += 2 if self.position_encoder else 0
            pos_encoder_input_channels += 2 if self.position_encoder_im1 else 0
            self.pos_encoder_block = PreActBlock(
                pos_encoder_input_channels, self.pos_encoder_channels)
            self.num_out_layers += self.pos_encoder_channels

        # create dustbin learnable parameters
        if self.dustbin:
            self.bin_score = torch.nn.parameter.Parameter(100*torch.ones(1, 1, 1))
            self.bin_feature = torch.nn.parameter.Parameter(
                torch.zeros(1, volume_channels, 1), requires_grad=False)

    def forward(self, vol0, vol1):
        assert vol0.shape == vol1.shape, 'Feature volumes shape must match'

        # reshape feature volumes
        B, D, H, W = vol0.shape
        vol0 = vol0.view(B, D, H * W)
        vol1 = vol1.view(B, D, H * W)

        # normalise features before dot product
        if self.normalise_dot_prod:
            vol0 = torch.nn.functional.normalize(vol0, dim=1)
            vol1 = torch.nn.functional.normalize(vol1, dim=1)

        # computes correlation volume
        # softmax along last dimension -> for each feature in vol0, gets a discrete distribution over vol1 features
        if self.cv_half_channels:
            cvolume = torch.bmm(vol0[:, :D//2].transpose(1, 2), vol1[:, :D//2])  # [B, H*W, H*W]
        else:
            cvolume = torch.bmm(vol0.transpose(1, 2), vol1)  # [B, H*W, H*W]

        # add learned bin score to correlation volume
        # add learned bin feature to vol1
        if self.dustbin:
            cvolume = torch.cat((cvolume, self.bin_score.repeat(B, 1, H*W)), dim=1)
            cvolume = torch.cat((cvolume, self.bin_score.repeat(B, H*W+1, 1)), dim=2)
            vol1 = torch.cat((vol1, self.bin_feature.repeat(B, 1, 1)), dim=2)

        # normalise correlation volume using softmax
        cvolume = torch.softmax(cvolume, dim=2)            # [B, H*W (+1), H*W (+1)]

        # warps vol1 using feature volume
        vol1w = torch.bmm(vol1, cvolume.transpose(1, 2))  # [B, D, H*W (+1)]
        if self.dustbin:
            vol1w = vol1w[:, :, :-1]                        # [B, D, H*W]

        cat_volumes = [vol0, vol1w]

        # adds u,v channels showing *average* position of the most prominent features
        if self.position_encoder:
            u = torch.linspace(-1, 1, H).to(vol0.device)
            v = torch.linspace(-1, 1, W).to(vol0.device)
            uu, vv = torch.meshgrid(u, v)
            grid = torch.stack([uu, vv], dim=0).view(2, H * W)  # [2, H*W]
            grid = grid.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, H*W]
            pos_encoder = torch.bmm(grid, cvolume[:, :H*W, :H*W].transpose(1, 2))  # [B, 2, H*W]
            cat_volumes.append(pos_encoder)
            if self.position_encoder_im1:
                cat_volumes.append(grid)

            # upsamples encoder features to multiple channels
            if self.pos_encoder_channels > 0:
                pos_encoder_features = torch.cat(
                    [pos_encoder, grid],
                    dim=1) if self.position_encoder_im1 else pos_encoder
                pos_encoder_features = pos_encoder_features.view(B, -1, H, W)
                pos_encoder_features = self.pos_encoder_block(pos_encoder_features)
                pos_encoder_features = pos_encoder_features.view(B, -1, H * W)
                cat_volumes.append(pos_encoder_features)

        # adds single channel showing *highest* score of a given feature vector in the other map
        # could help show the confidence in the matching (if max_score is low means multiple similar features)
        if self.max_score_channel:
            max_score = torch.max(cvolume, dim=2, keepdim=True)[
                0].transpose(1, 2)[..., :H*W]  # [B, 1, H*W]
            cat_volumes.append(max_score)

        # if cv_out_layers > 0, adds a 'reduced' correlation layer representation into the global volume
        if self.cv_out_layers > 0:
            cvolume_resized = cvolume[:, :H*W, :H*W].view(B, H*W, H, W)
            cvolume_reduced = self.CV_block(cvolume_resized)
            cvolume_reduced = cvolume_reduced.view(B, -1, H * W)
            cat_volumes.append(cvolume_reduced)

        agg_volume = torch.cat(cat_volumes, dim=1).view(B, -1, H, W)
        return agg_volume

class DeepResBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        bn = True
        self.avg_pool = True
        self.resblock1 = PreActBlock(in_channels, 64, stride=2)
        self.resblock2 = PreActBlock(64, 128, stride=2)
        self.resblock3 = PreActBlock(128, 256, stride=2)
        self.resblock4 = PreActBlock(256, 512, stride=2)

    def forward(self, feature_volume):
        B = feature_volume.shape[0]

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        if self.avg_pool:
            x = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        x = x.view(B, -1)
        return x

class DirectDeepResBlockMLP(DeepResBlock):
    """
    Direct R,t estimation using continuous 6D rotation representation from
    On the Continuity of Rotation Representations in Neural Networks
    https://arxiv.org/pdf/1812.07035.pdf
    """

    def __init__(self, in_channels):
        super().__init__(in_channels)

        self.mlp = torch.nn.Sequential(
            *[
                torch.nn.Linear(512, 256, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 6+3, bias=True)
            ])

    def forward(self, feature_volume):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        out = self.mlp(x).view(B, 9)

        R = rotation_6d_to_matrix(out[:, :6])
        t = out[:, 6:]
        return R, t

class Mapfree(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        model_configs = configs['model_configs']
        self.equivariance_mode = model_configs.get('equivariance_mode', False)
        self.encoder = ResUNet(model_configs['encoder'])
        self.aggregator = CorrelationVolumeWarping(self.encoder.num_out_layers)
        self.normalize_T = model_configs.get('normalize_T', False)
        self.decoder = DirectDeepResBlockMLP(self.aggregator.num_out_layers)
    
    def forward(self, inputs_data, mode='train'):
        x = inputs_data['images']  # B, 2, 3, H, W
        vol0 = self.encoder(x[:,0])
        vol1 = self.encoder(x[:,1])
        global_volume = self.aggregator(vol0, vol1)
        R1, t1 = self.decoder(global_volume)
        if self.normalize_T:
            t1 = F.normalize(t1, p=2, dim=-1)

        if self.equivariance_mode:
            global_volume2 = self.aggregator(vol1, vol0)
            R2, t2 = self.decoder(global_volume2)
            if self.normalize_T:
                t2 = F.normalize(t2, p=2, dim=-1)

            R2 = R2.transpose(-1, -2)
            R1tR2 = torch.bmm(R1.transpose(-1,-2), R2)
            R1tR2_sqr = axis_angle_to_matrix(matrix_to_axis_angle(R1tR2)/2)
            R = torch.bmm(R1, R1tR2_sqr)

            t = (t1-torch.bmm(R, t2.unsqueeze(-1)).squeeze(-1))/2
            if self.normalize_T:
                t = F.normalize(t, p=2, dim=-1)

            return {'rotation': R, 'translation': t}
        else:
            return {'rotation': R1, 'translation': t1}
            