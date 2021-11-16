# Deep Learning Image Processing Template (DLIP)
This repository is intended to be used as a template for image processing projects with focus on deep learning. Functionalities often used in such projects are already provided. This template is implemeneted in [PyTorch](https://pytorch.org/) and more specifically [PyTorch Lightning](https://www.pytorchlightning.ai/).

## Project Structure
Overview of this repository:
```
.
├── DLIP
│   ├── data #  Contains the defined datasets as PyTorch Lightning DataModules & Datasets.   
│   ├── experiments #  Contains experiment configurations as yaml files.
│   ├── models #  Contains the defined models as PyTorch Modules.    
│   ├── objectives #  Contains the defined objectives as PyTorch Modules.
│   ├── scripts #  Contains the training and inference scripts.
│   └── utils #  Contains utils functions, which can be used by all modules.
```

The training (`DLIP/scripts/train.py`) and inference script (`DLIP/scripts/inference.py`) are configured by the defined experiments (`DLIP/experiments`) and  utilize the defined datamodules (`DLIP/data`), models (`DLIP/models`) and objectives (`DLIP/objectives`).

## Install
### Prerequisite
- Python == 3.8.5
- Pip == 21.2.4
### Conda Environment
`conda create --name YOUR_ENV_NAME python=3.8.5`
### Pip Installation
1. Run `pip install -e .`
