#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4   #match with gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4
#SBATCH --account=begd-dtai-gh
#SBATCH --job-name=HP_optim
#SBATCH --time=24:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --output=/dev/null # to avoid double logging to slurm output file


# Organizing logging
# --------- Create timestamped output directory ----------
START_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="output_files/run_${START_TIMESTAMP}"
mkdir -p "$RUN_DIR"

# --------- Redirect stdout and stderr to log file -------
exec > >(tee -a "${RUN_DIR}/slurm-dtai-${SLURM_JOB_ID}.out") 2>&1

echo "Job started at $START_TIMESTAMP"
echo "Writing logs to $RUN_DIR"
# -----------------------------------------------------------------

module load python
module load python/anaconda3
module list  


total_gpus=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))
lr_init_scaled=$(echo "0.001 * $total_gpus" | bc -l)                   #$total_gpus gives the number of gpus in case lr scaling is desired

srun $(which python) train_torch.py --batch_size 32 \
                                    --data_dir "/work/nvme/begd/vsouzaramos/low_mass" \
                                    --checkpoint_dir "/projects/bdoy/vsouzaramos/final_dataset_validation/Wavenet_torch/" \
                                    --noise_dir "/projects/begd/vsouzaramos/new_data/processed_noise/" \
                                    --num_workers 8 \
                                    --num_nodes $SLURM_JOB_NUM_NODES \
                                    --lr_init 0.001

END_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Job ended at $END_TIMESTAMP"