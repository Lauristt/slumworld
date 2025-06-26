#!/bin/bash

# Collect config file to use (full path) from the command line and train a model on a single gpu
# MAKE SURE THAT the config file POINTS to the CORRECT dataset.csv 
CONFIG=${1:-"~/data/slumworld/slumworldML/config/base_pan_Bobo_gpu1.yml"}

if [[ $CONFIG == "h" ]] || [[ $CONFIG == "help" ]]; then
    echo "Script to use for scheduling a single model training run using the slurm manager"
    echo ""
    echo "ARGS:"
    echo "   CONFIG     str, the config file to use - place it within \"\" [default:\"~/slumworldML/config/base_savio_minas.yml\"]"
    echo "USAGE:"
    echo "   To train a model using the my_base_config.yml config file"
    echo "   . schedule-train_standard.sh \"~/slumworld\/config\/my_base_config.yml"
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-train_standard.sh  h"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~\/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/data/slumworld/slumworldML/slurm/train.slurm

sbatch --export=CONFIG=$CONFIG,FOLDID=-1 \
       train.slurm