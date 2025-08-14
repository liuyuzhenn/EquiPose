GPU=1

outdir=outputs/Scene7

# evaluate models without EquiPose
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/8vit.pth -o cfg_path=./configs/models/eight_vit.yml -w $outdir/EightVit
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/Mapfree.pth -o cfg_path=./configs/models/mapfree.yml -w $outdir/Mapfree 
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/GRelPose.pth -o cfg_path=./configs/models/grel_pose.yml -w $outdir/GRelPose 

# evaluate models with EquiPose (without fine-tuning)
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/8vit.pth -o cfg_path=./configs/models/eight_vit.yml     -w $outdir/EightVit-E  -o equivariance_inference=true
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/Mapfree.pth -o cfg_path=./configs/models/mapfree.yml    -w $outdir/Mapfree-E  -o equivariance_inference=true
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/GRelPose.pth -o cfg_path=./configs/models/grel_pose.yml -w $outdir/GRelPose-E  -o equivariance_inference=true

# evaluate models fine-tuned with EquiPose 
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/8vit-F.pth -o cfg_path=./configs/models/eight_vit.yml     -w $outdir/EightVit-F  -o equivariance_inference=true
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/Mapfree-F.pth -o cfg_path=./configs/models/mapfree.yml    -w $outdir/Mapfree-F  -o equivariance_inference=true
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --dataset ./configs/datasets/7scene.yml  -o ckpt_path=./weights/GRelPose-F.pth -o cfg_path=./configs/models/grel_pose.yml -w $outdir/GRelPose-F  -o equivariance_inference=true

python src/evaluation/show_summary.py --folder $outdir