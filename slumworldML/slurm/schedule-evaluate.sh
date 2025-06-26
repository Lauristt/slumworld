#!/bin/bash

CONFIG=${1:-"~/slumworldML/config/evaluate.yml"}

if [[ $CONFIG == "h" ]] || [[ $CONFIG == "help" ]]; then
    echo "Script to use for scheduling a model evaluation and map reconstruction job using the slurm manager"
    echo ""
    echo "ARGS:"
    echo "   CONFIG     str, the config file to use - place it within \"\" [default:\"~/slumworldML/config/evaluation.yml\"]"
    echo "USAGE:"
    echo "   To train a model using the my_evaluation_config.yml config file"
    echo "   . schedule-evaluate.sh \"~/slumworld\/config\/my_evaluation_config.yml"
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-evaluation.sh  h"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~\/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/slumworldML/slurm/evaluate.slurm


sbatch  --export=CONFIG=$CONFIG \
        evaluate.slurm