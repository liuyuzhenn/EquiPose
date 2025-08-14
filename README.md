# EquiPose: Exploiting Permutation Equivariance for Relative Camera Pose Estimation

This repository provideds the code associated wth our paper:
[EquiPose: Exploiting Permutation Equivariance for Relative Camera Pose Estimation]()

## Dependencies

- pytorch==1.13.1
- numpy==1.21.5

## Datasets

- [ScanNet](http://www.scan-net.org/)
- [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [12Scenes](https://graphics.stanford.edu/projects/reloc/)

Download and prepare the above datasets according to official documents.

Then create symlinks from the downloaded datasets to `./data/XX`, e.g., `./data/ScanNet`

## Evaluation

Download the pretrained model weights from the [link](https://drive.google.com/drive/folders/1zZCR9t8ptP-Y527H3CA6PVfrq23WZOcd?usp=drive_link), and save to ./weights



### ScanNet

`bash src/scripts/evaluate_scannet.sh`

### 7Scenes

`bash src/scripts/evaluate_7scenes.sh`

### 12Scenes

`bash src/scripts/evaluate_12scenes.sh`

## Citation

```
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Yuzhen and Dong, Qiulei},
    title     = {EquiPose: Exploiting Permutation Equivariance for Relative Camera Pose Estimation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {1127-1137}
}
```