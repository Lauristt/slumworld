#!/bin/bash

# Collect the number of folds and the config file to use (full path) from the command line
# MAKE SURE THAT the config file POINTS to the CORRECT kfold_dataset.csv 
NFOLDS=${1:-3}
CONFIG=${2:-"~/slumworldML/config/base_savio_minas.yml"}

if [[ $NFOLDS == "h" ]] || [[ $NFOLDS == "help" ]]; then
    echo "Script to use for scheduling  k-fold cross validation in parallel using the slurm manager"
    echo ""
    echo "ARGS:"
    echo "   NFOLDS     int, the number of folds [default:3]"
    echo "   CONFIG     str, the config file to use - place it within \"\" [default:\"~/slumworldML/config/base_savio_minas.yml\"]"
    echo ""
    echo "USAGE:"
    echo "   To train 3-folds in parallel using the my_base_config.yml config file [foldID in config file will not be used]:"
    echo "   . schedule-train_kfold.sh 3 \"~/slumworld\/config\/my_base_config.yml"
    echo "   To display usage help:"
    echo "   . schedule-train_kfold.sh h"
    echo ""
    echo "N.B.: If you have used a non default location for the slumworld library (assumed to be located in ~/)"
    echo "make sure to modify line 26 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/slumworldML/slurm/train.slurm

for FOLD in $(seq 0 $(expr $NFOLDS - 1) );do 
    sbatch --export=FOLDID=$FOLD,CONFIG=$CONFIG \
           train.slurm 
done