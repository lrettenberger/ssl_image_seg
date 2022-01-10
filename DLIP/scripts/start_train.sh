export RESULT_DIR="/home/ws/sc1357/data/inst_seg_tests/"
export CFG_BASE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc038/cfg_bbbc038_general.yaml"
export CFG_FILE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc038/cfg_bbbc038_train.yaml"

python train.py --config_files "\
$CFG_BASE \
$CFG_FILE\
" \
--result_dir $RESULT_DIR