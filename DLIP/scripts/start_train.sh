export RESULT_DIR="/home/ws/sc1357/data/2021_11_03_CLL_Seeds/results"
export CFG_BASE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc039/general_configuration.yaml"
export CFG_FILE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc039/bbbc039_configuration_train.yaml"

python train.py --config_files "\
$CFG_BASE \
$CFG_FILE\
" \
--result_dir $RESULT_DIR