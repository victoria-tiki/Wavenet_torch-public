#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdoy-delta-gpu
#SBATCH --job-name=new_wavenet
#SBATCH --time=16:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --output=/projects/bdoy/vsouzaramos/new_WaveNet_training/slurm-%j.out   # Save the output with a unique job ID in the specified directory

module load python
module load anaconda3_gpu/23.9.0    #anaconda3_cpu/23.7.4
module list  
nvidia-smi

total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)  #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) new_train_torch.py --batch_size 32 \
                                    --data_dir "/projects/bdoy/vsouzaramos/kiet_data2" \
                                    --checkpoint_dir "/projects/bdoy/vsouzaramos/new_WaveNet_training/checkpoints" \
                                    --resume_model "/projects/bdoy/vsouzaramos/new_WaveNet_training/checkpoints" \
                                    --noise_dir '/scratch/bdao/victoria/WaveNet_data' \
                                    --n_channels 2 \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001
