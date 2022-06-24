import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure

from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

config_files = "/home/ws/sc1357/projects/devel/src/detectron/DLIP/experiments/configurations/base_cfg/cfg_inst_seg_base.yaml"

cfg_yaml = merge_configs(config_files)
experiment_name=cfg_yaml['experiment.name']['value']

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=None,
    config_name=None,
    disabled=True
)

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"],  checkpoint_path_str="/home/ws/sc1357/data/inst_seg_tests/first-shot/DetectronDataModule/Detectron2/0009/dnn_weights.ckpt")


import tifffile

img = tifffile.imread("/home/ws/sc1357/data/2022_DMA_Spheroid_Detection_split/train/samples/A945_spot_row_15_col05.tif")


import torch
model.model.eval()
model.model.training = False
img_dict = dict()
img_dict["image"] = torch.from_numpy(img).permute(2,0,1)
input = [img_dict]

res = model.model(input)

print(res)
