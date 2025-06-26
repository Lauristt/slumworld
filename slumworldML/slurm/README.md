## This directory contains bash scripts that can be used to schedule jobs on savio using SLURM
### The bash/slurm scripts execute [python scripts](../runners) which perform the underline tasks


To perform any particular task one only needs to execute the correspoing `schedule-TASK_NAME.sh`, where TASK_NAME is the name of the task to be scheduled, providing the path to a configuration.yml file (by default located in the `../config folder`) which will provide the script with the required (hyper)parameters. 

Each bash script comes with help docstrings and usage examples. To display help on a bash script, execute the script with the flag `-h`, 
e.g. `. schedule_tile_kfold.sh -h`. 
For `*.slurm` scripts there is no interactive help functionality, however, there are extensive comments within the code.

The following sceduling scripts are provided (for more information inspect the script help strings and comments and also the [python scripts](../runners/)):  
1. `schedule-evaluate.sh`: For evaluating a trained model on a test set (requires that the test set has labels). Required parameters are provided in the [`evaluate.yml`](../config/evaluate.yml) configuration file.
2. `schedule-inference.sh`:  For performing inference (prediction) using a trained model on a (potentially new) dataset (does not require that the new dataset has labels). Required parameters are provided in the [`inference.yml`](..config/inference.yml) configuration file.
3. `schedule-tile_standard.sh`:  For tiling one or more satellite images on parallel, splitining them into train-validation and test sets and calculating balanging statistics (saved in the `dataset.csv` file). It can work on simple satellite images (i.e. without any labels), on pairs of images (i.e. satellite image plus labels) and can also accept a third image, a mask, to mask out areas that are not of interest (hence to economise on compute). Required parameters for each satellite image (duplet or triplet) that will be processed are provided in a [`tile.yml`](../config/tile.yml) configuration file. The script expects `\n` delimited file that lists the location (i.e. full path to each of the) various `tile.yml` files (on for each satellite image) that will be processed. A sample such list is [provided here](../config/tile_config_list.lst).
4. `schedule-tile_kfold.sh`:  Same as above, but spliting into __k__ different sets, using __kfold cross-validation__. A sample such list is [provided here](../config/tile_config_kfold_list.lst).
5. `schedule-tile_with_overlap.sh`: For tiling images for inference using 50% overlap between tiles and an extended padding (of 1/4 tile) along each side of the image. The overlap (together with the extended side padding) ensures that predictions for pixels in the central 1/2 part of each tile (i.e. between 1/4 tile to 3/4 tile, in both axis) will have "seen" context of, at least, 1/4 tiles and hence are likely to better deal with edge effects. When running evaluation with this tiling one should always use the relevant methods of evaluate.py. The script expects a configuration file with the required parameters. A sample configuration file is [provided here](../config/tile_with_overlap.yml).
6. `schedule-train_standard.sh`: For training a model on a dataset. Required parameters are provided in the [`base_savio.yml`](../config/base_savio_minas.yml) configuration file.
7. `schedule-train_kfold.sh`: Same as above but for training __k__ models on parallel, using a k-fold cross-validation dataset (i.e. a `dataset_kfold.csv` file).
8. `schedule-hyperparamter_search.sh`: Perform a hyper-parameter search by scheduling (in parallel) many jobs with different combinations of hyperparameters. Required parameters combinations are provided in (tab separated form) in file [hyperparameter_search.lst](./config/hyperparamter_search.lst). Each line in this file should include the configuration file that will be used (full or relative path), the fold_id (-1 for a standard run) and the parameters that will be varied (see the command line arguments from the command line parser in train.py).

Before using any of the schedulling scripts you have to manually __set the location of the slurm logs__ to your one home directory __in each script ending with `*.slurm` suffix__.

You should change all --output \[-o\] and --error \[-e\] logs lines, such as:
```
# location of logs (make sure directory exists and is writable)
#SBATCH -o /global/home/users/mksifaki/logs/version-%j-logs.out
#SBATCH -e /global/home/users/mksifaki/logs/version-%j-errors.err
```
to __your desired logging location__ (this has to be under your username so that you have the right to write to it).  


__N.B.__: Other than modifying the parameters in the configuration.yml files, the user should not need to do anything else than run the relevant `schedule-TASK_NAME.sh` scripts.
