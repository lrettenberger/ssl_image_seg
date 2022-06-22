import os
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_transforms import load_transforms
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
import matplotlib.pyplot as plt

import tifffile
from DLIP.utils.loading.load_model import load_model
import numpy as np
import torch

args = parse_arguments()
#config_files, ckpt_path = args["config_files"],  args["ckpt_path"]

config_files = "/home/ws/sc1357/projects/devel/src/dma-spheroid-bf/DLIP/experiments/configurations/dma_sph_bf/cfg_dma_sph_bf_inf.yaml"
ckpt_path = "/home/ws/sc1357/data/dma_spheroid_new/first-shot/BaseInstanceSegmentationDataModule/UnetInstSegSupervised/0000/dnn_weights.ckpt"

cfg_yaml = merge_configs(config_files)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=None,
    config_name=None,
    disabled=True
)

parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], checkpoint_path_str=ckpt_path)
model.eval()
model.cuda()

img_path = "/home/ws/sc1357/data/2022_DMA_Spheroid_Detection_split/train/samples/refine_spot_row_03_col07.tif"
img = tifffile.imread(img_path)[:,:,0]

_,_,test_trafos = load_transforms(parameters_splitted["data"])

input_tensor,_,_ = test_trafos[0](img)

pred = model(input_tensor.unsqueeze(0).cuda()).detach().cpu().numpy() 

plt.imshow(pred.squeeze())
plt.show()
dummy_input = torch.randn(1, 1, 128, 128, device="cuda")


input_names = [ "input_1" ]
output_names = [ "output_1" ]

torch.onnx.export(model, dummy_input, "2022_04_21_dma_spheroid_bf.onnx", verbose=True, input_names=input_names, output_names=output_names)


# import onnxruntime as ort

# model_path = "/home/ws/sc1357/projects/devel/src/dma-spheroid-bf/dma_spheroid_bf.onnx"

# providers = [
#     ('CUDAExecutionProvider', {
#         'device_id': 0,
#         'arena_extend_strategy': 'kNextPowerOfTwo',
#         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#         'cudnn_conv_algo_search': 'EXHAUSTIVE',
#         'do_copy_in_default_stream': True,
#     }),
#     'CPUExecutionProvider',
# ]

# session = ort.InferenceSession(model_path, providers=providers)

# img_path = "/home/ws/sc1357/data/2022_DMA_Spheroid_Detection_split/unlabeled/samples/spot_row_43_col07.tif"
# img = tifffile.imread(img_path)[:,:,0]

# _,_,test_trafos = load_transforms(parameters_splitted["data"])

# input_tensor,_,_ = test_trafos[0](img)

# outputs = session.run(
#     None,
#     {"input_1": input_tensor.unsqueeze(0).numpy()},
# )

# plt.imshow(outputs[0].squeeze())
# plt.show()