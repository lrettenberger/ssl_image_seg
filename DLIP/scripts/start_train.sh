export RESULT_DIR="/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation"
export CFG_FILE="/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/ssl/breastcancer/train.yaml"

python ssl_breast_train.py --config_files "\
$CFG_FILE\
" \
--result_dir $RESULT_DIR