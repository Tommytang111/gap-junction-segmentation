#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=128G
#SBATCH --time=10:0:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=def-mzhen
module purge
module load scipy-stack gcc cuda opencv
source ~/py10/bin/activate
wandb offline
python train.py 
