#!/bin/bash -l
# Use the current working directory
#SBATCH -D ../slurm
# Reset environment for this job.
# SBATCH --export=NONE
#Specify the GPU partition
#SBATCH -p gpu
#SBATCH --exclude=gpu030
#Specify the number of GPUs to be used
#SBATCH --gres=gpu:4
# Define job name
#SBATCH -J ImgNet
# Alocate memeory per core
#SBATCH -A bdliv03
#SBATCH --mem-per-cpu=32000M
# Setting maximum time days-hh:mm:ss]
#SBATCH -t 34:00:00
# Setting number of CPU cores and number of nodes
#SBATCH -n 16 -N 1

# Load modules
module load cuda/10.1.243

# Change conda env
conda activate pytorch

cd ../classification

# torchrun --nproc_per_node=4 train.py --model simple_vit --batch-size 512 --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ra --epochs 200 --weight-decay 0.02 --norm-weight-decay 0.0 --mixup-alpha 0.8 --cutmix-alpha 1.0 --dset_name=ImageNet --output-dir ../experiments/vit_inet_e200_raug_gumbel_se --opt adamw --amp --train-crop-size 192 --val-resize-size 232 --use_gumbel_se

# torchrun --nproc_per_node=4 train.py --dset_name=ImageNet --model resnet50 --output-dir ../experiments/r50_inet_e100_aaug -b 64 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --lr 0.256 --epochs 100 --mixup 0.2 --wd 6.103515625e-05 --auto-augment imagenet --momentum 0.875  --amp --train-crop-size 192 --val-resize-size 232


# torchrun --nproc_per_node=4 train.py --model s_vit --batch-size 512 --lr 1.2e-3 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --epochs 280 --weight-decay 0.05  --dset_name=ImageNet --output-dir ../experiments/test --criterion simmim --opt adamw --amp --transformer-embedding-decay 0 --bias-weight-decay 0 --norm-weight-decay 0  --train-crop-size 192 --val-resize-size 192 --val-crop-size 192 --no-val --attn sigmoid --ss_loss


torchrun --nproc_per_node=4 train.py --dset_name=ImageNet --model se_resnet101 --output-dir ../experiments/imagenet_full_se_r101_cosine_scheduler_mixup_aug -b 64 --lr-scheduler cosineannealinglr --reduction mean --lr 0.256 --epochs 100 --mixup-alpha 0.2 --wd 6.103515625e-05 --momentum 0.875 --auto-augment imagenet --lr-warmup-epochs 5 --amp