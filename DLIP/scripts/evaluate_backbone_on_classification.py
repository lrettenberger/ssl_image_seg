from cv2 import COLOR_BGR2RGB
import matplotlib

from DLIP.utils.evaluation.accuracy_with_dirs import calculate_accuracies
from DLIP.utils.evaluation.calculate_cka import calculate_cka
from DLIP.utils.evaluation.nearest_neighbour_retrival import get_nearest_neighbour
from DLIP.utils.evaluation.plot_2_pca import plot_2_pca
matplotlib.use('Agg')

import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch
from tqdm import tqdm

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer

directory = 'xray'

checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/NIHChestXrayDataModule/ResnetClassifier/0017/dnn_weights.ckpt' 
ref_checkpoint_path = checkpoint_path  

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

args = parse_arguments()
config_files, result_dir = args["config_files"], args["result_dir"]

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

# set wandb to disabled
cfg_yaml['wandb.mode'] = {'value' : 'disabled'}
# Encoder should not be frozen for evaluation
if 'model.params.encoder_frozen' in cfg_yaml:
    cfg_yaml['model.params.encoder_frozen'] = {'value' : False}

experiment_dir, config_name = prepare_directory_structure(
    base_path=base_path,
    experiment_name=experiment_name,
    data_module_name=cfg_yaml['data.datamodule.name']['value'],
    model_name=cfg_yaml['model.name']['value']
)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=experiment_dir,
    config_name=config_name
)
logging.warn(f"Working Dir: {os.getcwd()}")
seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], 
    checkpoint_path_str=checkpoint_path                 
)

ref_model  = load_model(parameters_splitted["model"], 
    checkpoint_path_str=ref_checkpoint_path                 
)

data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(parameters_splitted['train'], experiment_dir, wandb.run.name, data)


print('CALCULATING ACCURACIES')
calculate_accuracies(
    num_classes=1,
    channels=1,
    directory=directory,
    model=model,
    data=data,
)

print('CALCULATING NEAREST NEIGHBOURS')
get_nearest_neighbour(
    num_classes=1,
    channels=1,
    directory=directory,
    model=model,
    data=data,
)

print('CALCULATING 2 PCA')
plot_2_pca(
    num_classes=1,
    directory=directory,
    model=model,
    data=data,
)

# calculate_cka(
#     data=data,
#     directory=directory,
#     model=model,
#     ref_model=ref_model
# )
