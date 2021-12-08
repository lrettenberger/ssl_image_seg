import os
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

import tifffile

# parameter
window_size     = 512
resize_factor   = 1.0
ending_prediction = "_pred_raw"
relevant_dirs   = ["Hoechst", "Calcein", "PI"]


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


for subdir in tqdm(os.listdir(args["raw_data_path"]), desc="Inference folder-wise"):
    if subdir in relevant_dirs:
        results_dir     = os.path.join(args["raw_data_path"], subdir,"results")
        pred_raw_dir    = os.path.join(results_dir, "pred_raw")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if not os.path.exists(pred_raw_dir):
            os.makedirs(pred_raw_dir)

        for file in tqdm(os.listdir(os.path.join(args["raw_data_path"], subdir)), desc="Doing inference"):
            if not os.path.isfile(os.path.join(args["raw_data_path"], subdir, file)):
                continue
            img = cv2.imread(os.path.join(args["raw_data_path"], subdir, file),-1)
            # plt.imshow(img, cmap='gray') 
            # plt.show()

            img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)

            # plt.imshow(img, cmap='gray') 
            # plt.show()
            img = (65535 * img).astype(np.uint16)


            if resize_factor<1:
                scale_percent = resize_factor* 100
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim_new = (width, height)
                dim_old = (img.shape[1], img.shape[0])
                img = cv2.resize(img, dim_new, interpolation = cv2.INTER_CUBIC)

            input_tensor,_,_ = test_trafos[0](img)

            prediction = (window_inference(model,input_tensor, n_classes=1, window_size=512).squeeze().numpy())

            if resize_factor<1:
                prediction = cv2.resize(prediction, dim_old, interpolation = cv2.INTER_NEAREST)

            tifffile.imwrite(os.path.join(pred_raw_dir, file.replace(".tif", f"{ending_prediction}.tif")), prediction.astype("float32"))