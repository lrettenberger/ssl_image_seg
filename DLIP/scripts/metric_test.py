import os
from sklearn import metrics
import wandb
from tqdm import tqdm
import logging
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_transforms import load_transforms
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.helper_functions.window_inference import window_inference

import matplotlib.pyplot as plt

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor
from DLIP.utils.metrics.inst_seg_metrics import remap_label, get_fast_aji,get_dice_2, get_dice_1, get_fast_dice_2, get_fast_aji_plus
from skimage.color import label2rgb

# parameter
window_size     = 512

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

args = parse_arguments()
config_files, ckpt_path = args["config_files"], args["ckpt_path"]

cfg_yaml = merge_configs(config_files)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    disabled=True,
    experiment_dir=None,
    config_name=None,
)

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], checkpoint_path_str=ckpt_path)
_, _, test_trafos = load_transforms(parameters_splitted["data"])

model.cuda()
model.eval()

dist_map_processor = DistMapPostProcessor()

img = cv2.imread("/home/ws/sc1357/data/datasets/BBBC038_clean/train/samples/406_2.png",-1)
img_org = img.copy()
gt_label = cv2.imread("/home/ws/sc1357/data/datasets/BBBC038_clean/train/labels/406_2.png",-1)

# plt.imshow(img, cmap='gray') 
# plt.show()

img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)

# plt.imshow(img, cmap='gray') 
# plt.show()
img = (65535 * img).astype(np.uint16)

input_tensor,_,_ = test_trafos[0](img)

prediction = (window_inference(model,input_tensor, window_size=512).squeeze().numpy())

# plt.imshow(prediction, cmap='gray') 
# plt.show()

pred_inst = dist_map_processor.process(prediction, img)

fig, ax = plt.subplots(1,3)
ax[0].imshow(img_org, cmap='gray') 
ax[1].imshow(label2rgb(gt_label, bg_label=0), cmap='gray') 
ax[2].imshow(label2rgb(pred_inst, bg_label=0), cmap='gray') 
plt.show()

metrics_dict = {
    "DSC_1": get_dice_1,
    "DSC_2": get_dice_2,
    "DSC_2_fast": get_fast_dice_2,
    "AJI": get_fast_aji,
    "AJI+": get_fast_aji_plus,
}

for (key,value) in metrics_dict.items():    
    print(f"{key}: {value(remap_label(gt_label), remap_label(pred_inst))}")