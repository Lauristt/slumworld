#!/bin/bash

TILE_CONFIG=${1:-"~/slumworldML/config/tiling_with_overlap.yml"}

if [[ $TILE_CONFIG == "h" ]] || [[ $TILE_CONFIG == "help" ]]; then
    echo "Script to use for scheduling tiling raw images with 50% tile overlap and extended (1/4th tile) top+left padding for inference."
    echo "The script should be supplied with the path to the relevant configuration file."
    echo ""
    echo "ARGS:"
    echo "   CONFIG     str, the \\n separated file with the paths to the individual tile.yml files to use for tiling"
    echo "              it is reccomended to place this within \"\" [default:\"~/slumworldML/config/tile_with_overlap.yml\"]"
    echo "USAGE:"
    echo "   To train a model using the a different tile_with_overlap.yml"
    echo "   . schedule-tile_with_overlap.sh \"\/path\/to\/tile_with_overlap.yml"
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-tile_with_overlap.sh  h"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~\/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/slumworldML/slurm/tile_with_overlap.slurm


sbatch --export=TILE_CONFIG=$TILE_CONFIG \
       tile_with_overlap.slurm