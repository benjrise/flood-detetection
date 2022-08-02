#!/bin/bash -l
# Use the current working directory for output
#SBATCH -D ./
# Reset environment for this job.
# SBATCH --export=NONE
#Specify the partition, either gpu or cpu
#SBATCH -p gpu
#Specify the number of GPUs to be used
#SBATCH --gres=gpu:1
# Define job name
#SBATCH -J spacenet8
# Setting maximum time days-hh:mm:ss]
#SBATCH -t 72:00:00
# Setting number of CPU cores and number of nodes
#SBATCH --mem-per-cpu=23000M
#SBATCH -n 4 -N 1
# Load modules
module purge
module load apps/singularity/3.5.0-rc.2 

singularity run --nv spacenet.sif train_foundation.sh

