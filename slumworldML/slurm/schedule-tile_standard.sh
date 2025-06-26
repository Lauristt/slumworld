#!/bin/bash


TILE_CONFIG_LIST=${1:-"~/slumworldML/config/tile_config_list.lst"}

if [[ $TILE_CONFIG_LIST == "h" ]] || [[ $TILE_CONFIG_LIST == "help" ]]; then
    echo "Script to use for scheduling tiling raw images (using a different tiling specification for each one) in parallel."
    echo "The specification for each file should be defined in a separate tile.yml file."
    echo "The script should be supplied with a a new line delimited list of the various tile.yml files."
    echo ""
    echo "ARGS:"
    echo "   CONFIG     str, the \\n separated file with the paths to the individual tile.yml files to use for tiling"
    echo "              it is reccomended to place this within \"\" [default:\"~/slumworldML/config/tile_config_list.lst\"]"
    echo "USAGE:"
    echo "   To train a model using the a different tile_config_list2.lst"
    echo "   . schedule-tile_standard.sh \"\/path\/to\/tile_config_list.lst"
    echo ""
    echo "   To display usage help:"
    echo "   . schedule-tile_standard.sh  h"
    echo ""
    echo "If you have used a non default location for the slumworld library (assumed to be located in ~\/)"
    echo "make sure to modify line 24 of the script accordingly (to enable execution of the file by the current user)."
    return 0
fi

chmod +x ~/slumworldML/slurm/tile_standard.slurm


sbatch --export=TILE_CONFIG_LIST=$TILE_CONFIG_LIST \
       tile_standard.slurm