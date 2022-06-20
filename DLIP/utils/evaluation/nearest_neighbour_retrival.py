from pathlib import Path
from tkinter import N
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def nearest_neighbors(value, array):
    return np.argsort(np.array([np.linalg.norm(value-x) for x in array]))

def get_nearest_neighbour(num_classes,channels,directory,model,data, class_type='resnet_classifier'):
    nbr_neighbors=10
    samples = 30
    latent_dim = None
    if num_classes == 1:
        print('not implemented ...')
    else:
        Path(f"{directory}/nearest_neighbours").mkdir(parents=True, exist_ok=True)
        
        composition = ModuleList()
        if class_type == 'resnet_classifier':
            composition.append(model.composition[0])
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            latent_dim = 2048
            
        
        embeddings = np.zeros((0,latent_dim))
        y_trues = np.zeros((0,num_classes))
        xs = np.zeros((0,256,256,channels))
        for batch in tqdm(data.test_dataloader()):
            x,y_true = batch
            y_pred = x
            for item in composition:
                y_pred = item(y_pred.to('cuda'))
            y_pred = y_pred.squeeze()
            embeddings = np.concatenate((embeddings,y_pred.cpu().detach()),axis=0)
            y_trues = np.concatenate((y_trues,y_true.detach().cpu()),axis=0)
            xs = np.concatenate((xs,x.permute(0,2,3,1).cpu().detach()*255),axis=0)
        nearest_indices = []
        for i in tqdm(range(len(embeddings))):
            nearest_indices.append(nearest_neighbors(embeddings[i],embeddings))
        nearest_indices = np.array(nearest_indices)
        for i in tqdm(range(min([len(nearest_indices),samples]))):
            neighbourhood = nearest_indices[i]
            base_dir = f"{directory}/nearest_neighbours/{neighbourhood[0]}"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            ref_img = xs[neighbourhood[0]]
            plt.imshow(ref_img.astype(np.uint8))
            plt.title(f'Reference {np.argmax(y_trues[neighbourhood[0]])}')
            plt.savefig(f'{base_dir}/0_reference_class_{np.argmax(y_trues[neighbourhood[0]])}.png')
            plt.close()
            for neighbour in range(1,nbr_neighbors+1):
                n_index = neighbourhood[neighbour]
                plt.imshow(xs[n_index].astype(np.uint8))
                plt.title(f'class {np.argmax(y_trues[n_index])}')
                plt.savefig(f'{base_dir}/{neighbour}.png')
                plt.close()
        