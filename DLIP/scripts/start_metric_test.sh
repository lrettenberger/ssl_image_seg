export CFG_BASE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc038/cfg_bbbc038_general.yaml"
export CFG_FILE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc038/cfg_bbbc038_train.yaml"
export CKPT_PATH="/home/ws/sc1357/data/2021_11_03_CLL_Seeds/results/small-dataset-training/BaseInstanceSegmentationDataModule/UnetInstSegSupervised/0029/dnn_weights.ckpt"

python metric_test.py --config_files "\
$CFG_BASE \
$CFG_FILE\
" \
--ckpt_path $CKPT_PATH