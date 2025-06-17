#!/bin/bash
#SBATCH --mem-per-cpu=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu #gpuA100x4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdao-delta-cpu #bcdz-delta-gpu
#SBATCH --job-name=new_val_data_eval 
#SBATCH --time=4:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpus-per-node=4 
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --output=/projects/bdoy/vsouzaramos/new_WaveNet_training/inference/val_data_eval-%j.out   

module load python
module load anaconda3_cpu/23.7.4 #anaconda3_gpu/23.9.0    #anaconda3_cpu/23.7.4
#module load anaconda3_gpu/23.9.0
module list  # job documentation and metadata

echo "job is starting on `hostname`"
srun $(which python) new_val_data_eval_full.py