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
#SBATCH -t 48:00:00
# Setting number of CPU cores and number of nodes
#SBATCH -n 16 -N 1

# Load modules
module load cuda/10.1.243

# Change conda env
conda activate pytorch

cd ../classification

# torchrun --nproc_per_node=4 train.py --model se_resnet50 --batch-size 64 --lr 0.2 --lr-scheduler cosineannealinglr --lr-warmup-epochs 1 --lr-warmup-method linear --auto-augment --epochs 200 --weight-decay 0.0001  --mixup-alpha 0.2 --dset_name=imagenet_lt --classif_norm cosine --use_gumbel_se --output-dir ../experiments/se_r50_ilt_ce_mean_gumbel_se_wd1e-5 --amp

# torchrun --nproc_per_node=4 train.py --model se_resnet50 --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment imagenet --epochs 600 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --dset_name=imagenet_lt --output-dir ../experiments/r50_ilt_ce_mean_gumbel_se_wd1e-4_b256_e600_aa --classif_norm cosine --use_gumbel_se

# torchrun --nproc_per_node=4 train.py --model se_resnet50 --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment imagenet --epochs 600 --random-erase 0.1 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --dset_name=imagenet_lt --output-dir ../experiments/r50_ilt_ce_mean_gumbel_se_wd1e-4_b256_e600_aa_re --classif_norm cosine --use_gumbel_se

torchrun --nproc_per_node=4 train.py --model cb_resnext50_32x4d --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment imagenet --epochs 600 --random-erase 0.1 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4 --dset_name=imagenet_lt --output-dir ../experiments/cb_x50_ilt_ce_mean_wd1e-4_b256_e600_aa_re_ra_gumbel_se --classif_norm cosine --use_gumbel_se --amp