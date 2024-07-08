#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcbw-delta-gpu
#SBATCH --job-name=2channelcontd
#SBATCH --output="wavenet_output/2channelcontd.out.%j.%N.out"
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4

echo Loading Conda Environment
source activate wavenet
conda info --envs
echo $CONDA_DEFAULT_ENV
echo Starting Training 


total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)                   #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) train_torch_2channel.py --batch_size 32 \
                                    --data_dir "/scratch/bcbw/amatchev/data/WaveNet_data/combined_spin/" \
                                    --checkpoint_dir "/scratch/bcbw/amatchev/training/WaveNet_checkpoints/" \
                                    --resume_model "/scratch/bcbw/amatchev/training/WaveNet_checkpoints/old_checkpoints/2channel_epoch=29-val_loss=0.02447.ckpt" \
                                    --noise_dir "/scratch/bcbw/amatchev/data/WaveNet_data/Gaussian_Noise/" \
                                    --n_channels 2 \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001
