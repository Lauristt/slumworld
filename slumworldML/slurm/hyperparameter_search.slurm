#!/bin/bash

## Sample script for running a single hyper-parameter run of the pytorch model
#SBATCH --job-name=UNET_run_no_augmentations
#SBATCH --account=fc_slums
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

## Set number of nodes and cores per node; reserve GPU(s) and set compute mode
#SBATCH --nodes=1
#SBATCH --ntasks=1

## Numbers of cpu cores: must be 2x the number GPUs 
#SBATCH --cpus-per-task=2
###SBATCH --cpus-per-gpu=2


## Number of gpus
#SBATCH --gres=gpu:1    ### if you want more gpus put it here

## Set maximum walltime (HH:mm:ss), maximum is 72 hours
#SBATCH --time=72:00:00

## If you wish to restart your job upon termination with pytorch_lightning remove the ! from the next line
#!SBATCH --signal=SIGUSR1@90

## enable email notifications - uncomment below (remove the !) and add your email to introduce email notifications
#!SBATCH --mail-user=your.email@gmail.com
#!SBATCH --mail-type=ALL

#! location of logs (make sure directory exists and is writable)
#SBATCH -o /global/home/users/mksifaki/logs/version-%j-logs.out
#SBATCH -e /global/home/users/mksifaki/logs/version-%j-errors.err

# Load any kernel modules needed for execution
. ~/.bashrc

# Receive $CONFIG, $FOLDID from the environemnt (exported by the scheduling script)
# 1->FOLDID, the current fold to train on (setting the default value to -1, i.e. standard single fold split) 
# 2->CONFIG, the configuration file to use for training (setting the default to "~/slumworldML/config/base_savio_minas.yml")

# Activate your virtual environment
echo "Activating environment:" $(which conda)
conda activate torchenv 
echo Conda environment $CONDA_PREFIX activated
# Last chance to set permissions to directories
# ADD HERE ANY CUSTOM CONFIG FILE THAT YOU WILL BE USING
chmod -R +w ~/logs
chmod -R +rx ~/slumworldML
chmod -R +r ../config
chmod +rx ../runners/train.py
echo " "
echo "Job started on `date`"
echo "Number of credits on account at job start:"
echo $(check_usage.sh -a fc_slums)
echo " "

# actual script to execute (you can pass additional command line arguments here)

python ../runners/train.py $PARAMS


echo " "
echo "Job ended on `date`"
echo "Number of credits on account at job end:"
echo $(check_usage.sh -a fc_slums)
echo " "
echo " "
echo "Job used the following resources:"
echo $(scontrol show job $SLURM_JOB_ID)



