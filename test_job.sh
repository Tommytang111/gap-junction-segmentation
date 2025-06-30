#!/bin/bash
#SBATCH --job-name=test_job_train
#SBATCH --account=def-mzhen
#SBATCH --output=output_%j.txt
#SBATCH --time=1:00:00 #1hour
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --mail-user=tommytang111@hotmail.com
#SBATCH --mail-type=ALL

echo tommy
