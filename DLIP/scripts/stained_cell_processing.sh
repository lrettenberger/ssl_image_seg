export CFG_BASE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc039/general_configuration.yaml"
export CFG_FILE="/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/bbbc039/bbbc039_configuration.yaml"
export CKPT_PATH="/home/ws/sc1357/data/2021_11_03_CLL_Seeds/results/small-dataset-training/BaseInstanceSegmentationDataModule/UnetInstSegSupervised/0029/dnn_weights.ckpt"
export RAW_DATA_PATH="/home/ws/sc1357/data/2021_06_25_CLL_Cells/2021_12_03_CLL_Instance_Detection"

python inference.py --config_files "\
$CFG_BASE \
$CFG_FILE\
" \
--ckpt_path $CKPT_PATH \
--raw_data_path $RAW_DATA_PATH

python filter_predictions.py --raw_data_path $RAW_DATA_PATH 

python stained_cell_fusion.py --raw_data_path $RAW_DATA_PATH 