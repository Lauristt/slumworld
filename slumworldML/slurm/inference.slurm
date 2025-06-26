#!/bin/bash

# Sample script for running the pytorch model
#SBATCH --job-name=UNET_predict
#SBATCH --account=fc_slums
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

# Set number of nodes and cores per node; reserve GPU(s) and set compute mode
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Numbers of cpu cores: must be 2x the number GPUs 
#SBATCH --cpus-per-task=2
###SBATCH --cpus-per-gpu=2


# Number of gpus
#SBATCH --gres=gpu:1    ### if you want more gpus put it here

# Set maximum walltime (HH:mm:ss), maximum is 72 hours
#SBATCH --time=01:00:00

# enable email notifications - uncomment below (remove the !) and add your email to introduce email notifications
#!SBATCH --mail-user=your.email@gmail.com
#!SBATCH --mail-type=ALL

# location of logs (make sure directory exists and is writable)
#SBATCH -o /global/home/users/mksifaki/logs/inference-%j-logs.out
#SBATCH -e /global/home/users/mksifaki/logs/inference-%j-errors.err

# Load any kernel modules needed for execution
. ~/.bashrc
# Activate your virtual environment
echo $(which conda)
conda activate torchenv 
echo Conda environment $CONDA_PREFIX activated
# Last chance to set permissions to directories
chmod -R +w ~/logs
chmod -R +rx ~/slumworldML
chmod -R +w /global/scratch/users/mksifaki/output/experiments/SatelliteUnet/lightning_logs/
chmod +rx ../runners/inference.py
echo " "
echo "Job started on `date`"
echo "Number of credits on account at job start:"
echo $(check_usage.sh -a fc_slums)
echo " "

# actual script to execute (you can pass additional command line arguments here), CONFIG is exported from the scheduling script
python ../runners/inference.py -c $CONFIG

echo " "
echo "Job ended on `date`"
echo "Number of credits on account at job end:"
echo $(check_usage.sh -a fc_slums)
echo " "
echo " "
echo "Job used the following resources:"
echo $(scontrol show job $SLURM_JOB_ID)



