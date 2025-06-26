#!/bin/bash
## Sample script for tilling one or more images in parallel (e.g. MS and MD ) - or one image with different 
## parameters (e.g. different tiling size etc) in parallel (2 at a time) to economize on resources
## Reads the location of the tiling config files from a new-line separated file tile_configs_list.lst

#SBATCH --job-name=tiling_with_overlap
#SBATCH --account=fc_slums
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

## Set number of nodes and cores per node; reserve GPU(s) and set compute mode
#SBATCH --nodes=1
#SBATCH --ntasks=1

## Numbers of cpu cores: must be 2x the number GPUs 
#SBATCH --cpus-per-task=2

## Number of gpus
#SBATCH --gres=gpu:1    ### if you want more gpus put it here

## Set maximum walltime (HH:mm:ss), maximum is 72 hours
#SBATCH --time=24:00:00

## enable email notifications - uncomment below (remove the !) and add your email to introduce email notifications
#!SBATCH --mail-user=your.email@gmail.com
#!SBATCH --mail-type=ALL

## location of logs (make sure directory exists and is writable)
#SBATCH -o /global/home/users/mksifaki/logs/tiling_with_overlap-%j-logs.out
#SBATCH -e /global/home/users/mksifaki/logs/tiling_with_overlap-%j-errors.err

## Load any kernel modules needed for execution
. ~/.bashrc
module load gnu-parallel/2019.03.22
## Activate your virtual environment
echo Activating environment $(which conda)
conda activate torchenv 
echo Conda environment $CONDA_PREFIX activated
## Last chance to set permissions to directories
chmod -R +w ~/logs
chmod -R +rx ~/slumworldML
chmod -R +w ~/logs/
# chmod -R +wrx /global/scratch/users/mksifaki/data/
chmod +rx $TILE_CONFIG
chmod +rx ../runners/tile_with_overlap.py

echo " "
echo "Job started on `date`"
echo "Number of credits on account at job start:"
echo $(check_usage.sh -a fc_slums)
echo " "
parallel --citation
## actual script to execute
## TILE_CONFIG is exported from the scheduling script
python ../runners/tile_with_overlap.py -c $TILE_CONFIG

echo " "
echo "Job ended on `date`"
echo "Number of credits on account at job end:"
echo $(check_usage.sh -a fc_slums)
echo " "
echo " "
echo "Job used the following resources:"
echo $(scontrol show job $SLURM_JOB_ID)



