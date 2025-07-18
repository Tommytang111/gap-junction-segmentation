#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet_base
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=128G
#SBATCH --time=24:0:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tommytang111@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mzhen
module purge
module load gcc cuda opencv
source /home/tommy111/.gj_venv/bin/activate
wandb offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 /home/tommy111/projects/def-mzhen/tommy111/code/new/src/train.py 
