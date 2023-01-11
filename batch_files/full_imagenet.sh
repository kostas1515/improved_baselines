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

# torchrun --nproc_per_node=4 train.py --model simple_vit --batch-size 512 --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ra --epochs 200 --weight-decay 0.02 --norm-weight-decay 0.0 --mixup-alpha 0.8 --cutmix-alpha 1.0 --dset_name=ImageNet --output-dir ../experiments/vit_inet_e200_raug_gumbel_se --opt adamw --amp --train-crop-size 192 --val-resize-size 232 --use_gumbel_se

torchrun --nproc_per_node=4 train.py --model ca_vit --batch-size 512 --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ra --epochs 200 --weight-decay 0.02 --norm-weight-decay 0.0 --mixup-alpha 0.8 --cutmix-alpha 1.0 --dset_name=ImageNet --output-dir ../experiments/cait_inet_e200_raug --opt adamw --amp --find-unused-params