#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=begd-dtai-gh
#SBATCH --job-name=pre_hyperparameter_tuning  # Name of the job
#SBATCH --time=16:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --output=/dev/null # to avoid double logging to slurm output file

# --------- Create timestamped output directory ----------
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="/projects/bdoy/vsouzaramos/Hyperparameter_tests/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# --------- Redirect stdout and stderr to log file -------
exec > >(tee -a "${RUN_DIR}/slurm-dtai-${SLURM_JOB_ID}.out") 2>&1

echo "Job started at $TIMESTAMP"
echo "Writing logs to $RUN_DIR"

# --------- Load modules and run training ----------------
module load python
module load python/anaconda3
module list  
nvidia-smi

total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)  #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) train_torch.py --batch_size 32 \
                                    --data_dir "/projects/begd/vsouzaramos/new_data/data_highmass" \
                                    --noise_dir '/projects/begd/vsouzaramos/new_data/processed_noise/' \
                                    --checkpoint_dir "${RUN_DIR}/checkpoints" \
                                    --resume_model "${RUN_DIR}/checkpoints" \
                                    --n_channels 2 \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001 \
                                    --run_dir "$RUN_DIR" \
