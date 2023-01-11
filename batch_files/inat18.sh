#!/bin/bash -l
# Use the current working directory
#SBATCH -D ../slurm
# Reset environment for this job.
# SBATCH --export=NONE
#Specify the GPU partition
#SBATCH -p gpu
#Specify the number of GPUs to be used
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu030
# Define job name
#SBATCH -J inat18
# Alocate memeory per core
#SBATCH -A bdliv03
#SBATCH --mem-per-cpu=32000M
# Setting maximum time days-hh:mm:ss]
#SBATCH -t 48:00:00
# Setting number of CPU cores and number of nodes
#SBATCH -n 16 -N 1

# Load modules
module purge
module load cuda/10.1.243

# Change conda env
conda activate pytorch


cd ../classification

torchrun --nproc_per_node=4 train.py --model se_resnet50 --data-path ../../../datasets/  --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ra  --epochs 500 --random-erase 0.1 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4 --dset_name=inat18 --output-dir ../experiments/inat_se_r50_softmax_e500_cosine_mixup_wd1e-4_aug_gumbel_se --classif_norm cosine --use_gumbel_se