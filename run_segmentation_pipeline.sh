#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=unet_segment
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tommytang111@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mzhen

#Load modules
module purge
module load gcc/13.3 cuda/12.6 opencv
source /home/tommy111/.gj_venv/bin/activate

#Configure environment for parallel processing
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#Set number of threads for various libraries to match SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}

#Run segmentation pipeline
echo "Starting segmentation pipeline with ${SLURM_CPUS_PER_TASK} CPU cores and GPU: $(nvidia-smi -L)"
python3 /home/tommy111/projects/def-mzhen/tommy111/code/new/src/segment_dataset.py 
