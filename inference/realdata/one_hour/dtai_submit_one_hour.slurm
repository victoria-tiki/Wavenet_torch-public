#!/bin/bash
#SBATCH --mem-per-cpu=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4
#SBATCH --account=begd-dtai-gh
#SBATCH --job-name=new_one_hour 
#SBATCH --time=2:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --output=/projects/bdoy/vsouzaramos/new_WaveNet_training/one_hour/dtai-one_hour-%j.out   

module load python
module load python/anaconda3

module list  # job documentation and metadata

echo "job is starting on `hostname`"
srun $(which python) new_one_hour_noise.py