#!/bin/bash
#SBATCH --partition=haicore-gpu4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcel.schilling@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=first_shot
#SBATCH --constraint=LSDF

export DATASET="/lsdf01/lsdf/kit/iai/projects/iai-aida/Daten_Schilling/datasets/2022_DMA_Spheroid_Detection_split"
export RESULT_DIR="/lsdf01/lsdf/kit/iai/projects/iai-aida/Daten_Schilling/datasets/2022_DMA_Spheroid_Detection_split/results"

# remove all modules
module purge
module load compiler/intel/19.1 mpi/openmpi/4.0

# activate cuda
module load devel/cuda/11.2

# activate conda env
source /home/hk-project-sppo/sc1357/miniconda3/etc/profile.d/conda.sh
conda activate base_inst_seg

# move to script dir
cd /home/hk-project-sppo/sc1357/devel/base-instance-segmentation/DLIP/scripts

# update env vars
python update_env_vars.py --datatset_path $DATASET --result_dir $RESULT_DIR

# update cfg file
python write_parameter_to_yaml.py --cfg_file_path $CFG_FILE $CFG_ADD --datatset_path $DATASET

cd /home/hk-project-sppo/sc1357/devel/base-instance-segmentation/DLIP/utils/data_preparation/label_pre_processing

# generate dist maps
python generate_distance_maps.py --datatset_path $DATASET

# start train
python train.py --config_files $CFG_FILE --result_dir $RESULT_DIR