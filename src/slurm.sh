#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node

source /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/etc/profile.d/conda.sh

conda create --name gan
conda activate gan

python main.py xg Y 1 1 1

conda env remove -n gan