#!/bin/bash

#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --partition=gpu-medium
#SBATCH --array=0-2

SEED=$SLURM_ARRAY_TASK_ID

python tdmpc2_jax/train.py seed=$SEED >"${SEED}.out" 2>&1