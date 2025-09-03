#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=unet_segment
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=131072M
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tommytang111@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mzhen

#Load modules
module purge
module load gcc/13.3 cuda/12.6 opencv
source /home/tommy111/.gj_venv/bin/activate

wandb offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

#Run training script
python3 /home/tommy111/projects/def-mzhen/tommy111/code/new/src/segment_dataset.py 
