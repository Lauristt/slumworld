#!/bin/bash

# Collect the path to a (tab separated) list of k hyperparameter combinations from the command line and train k models on a single gpu
# each. MAKE SURE THAT all config files POINT to the CORRECT dataset.csv 
PARAM_FILE=${1:-"~/slumworldML/config/hyperparameter_search.lst"}

if [[ $PARAM_FILE == "h" ]] || [[ $PARAM_FILE == "help" ]]; then
    echo ""
    echo "Script to use for scheduling k single model training runs using the slurm manager and k-hyper-parameter combinations defined in "
    echo "tab separated list file: hyperparameter_search.lst"
    echo ""
    echo "ARGS:"
    echo "   PARAM_FILE     str, the config file to use - place it within \"\" [default:\"~/slumworldML/config/hyperparameter_search.lst\"]"
    echo "USAGE:"
    echo "   To train a model using the my_base_config.yml config file"
    echo "   . schedule-hyperparameter_search.sh \"~/slumworld/config/hyperparameter_search.lst\""
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-hyperparameter_search.sh  help"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    echo ""
    return 0
fi

chmod +x ~/slumworldML/slurm/hyperparameter_search.slurm
chmod u+x $PARAM_FILE
echo $PARAM_FILE

while IFS="" read -r p || [ -n "$p" ]; do 

    sbatch --export=PARAMS="$p" \
             hyperparameter_search.slurm
    echo "$p"
    
    done < $PARAM_FILE
