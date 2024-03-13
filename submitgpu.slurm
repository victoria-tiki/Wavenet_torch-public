#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcdz-delta-gpu
#SBATCH --job-name=wavenet
#SBATCH --time=16:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4


module load python
module load anaconda3_gpu/23.9.0    #anaconda3_cpu/23.7.4
module list  


total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)                   #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) train_torch.py --batch_size 32 \
                                    --data_dir "/scratch/bbke/victoria/WaveNet_data/combined_spin/" \
                                    --checkpoint_dir "/scratch/bbke/victoria/WaveNet_training/checkpoints/" \
                                    --resume_model "/scratch/bbke/victoria/WaveNet_training/checkpoints/" \
                                    --noise_dir "/scratch/bbke/victoria/WaveNet_data/Gaussian_Noise/" \
                                    --n_channels 3 \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001
