#!/bin/bash

CONFIG=${1:-"~/slumworldML/config/inference.yml"}

if [[ $CONFIG == "h" ]] || [[ $CONFIG == "help" ]]; then
    echo "Script to use for scheduling an inference run using the slurm manager"
    echo ""
    echo "ARGS:"
    echo "   CONFIG     str, the config file to use - place it within \"\" [default:\"~/slumworldML/config/inference.yml\"]"
    echo "USAGE:"
    echo "   To train a model using the my_inference_config.yml config file"
    echo "   . schedule-inference.sh \"~/slumworld\/config\/my_inference_config.yml"
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-inference.sh  h"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~\/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/slumworldML/slurm/inference.slurm

sbatch --export=CONFIG=$CONFIG \
       inference.slurm 