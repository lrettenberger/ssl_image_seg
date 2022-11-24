#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=152
#SBATCH --ntasks-per-core=2
#SBATCH --gres=gpu:full:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcel.schilling@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=dma
#SBATCH --mem=501600mb

export CFG_FILE="/home/iai/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/base_cfg/downstream/semantic/ds_derma.yaml"
export RESULT_DIR="/home/iai/sc1357/data/2022_11_24_ssl"
export SWEEPID="kit-iai-ibcs-dl/ssl_seg_derma/88c7x5t6"

NUM_GPUS=4
NUM_COUNTS=1

# remove all modules
module purge

# activate cuda
module load devel/cuda/11.2

# activate conda env
source /home/iai/sc1357/miniconda3/etc/profile.d/conda.sh
conda activate env_ssl

# move to script dir
cd /home/iai/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/scripts

# start train
if [[ $NUM_GPUS -eq 8 ]]
then
    CUDA_VISIBLE_DEVICES=0 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=1 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=2 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=3 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=4 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=5 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=6 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=7 wandb agent --count $NUM_COUNTS $SWEEPID 
else
    CUDA_VISIBLE_DEVICES=0 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=1 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=2 wandb agent --count $NUM_COUNTS $SWEEPID &
    CUDA_VISIBLE_DEVICES=3 wandb agent --count $NUM_COUNTS $SWEEPID 
fi

wait < <(jobs -p)