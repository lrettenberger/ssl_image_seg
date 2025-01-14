import tifffile
import glob
import os
import numpy as np
import cv2
import torch
from DLIP.utils.helper_functions.gray_level_check import gray_redundand

from DLIP.data.base_classes.base_dataset import BaseDataset


class BaseInstanceSegmentationDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_dir: str = "samples",
        labels_dir: str = "labels",
        labels_dmap_dir: str = "labels_dist_map",
        samples_data_format="tif",
        labels_data_format="tif",
        labels_dmap_data_format="tif",
        transforms = None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        label_suffix="_label",
        label_prefix="",
        instance_segmentation_head=False,
        **kwargs
    ):
        self.labels_available = labels_available
        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.labels_dir = labels_dir
        self.labels_dmap_dir = labels_dmap_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.labels_dmap_data_format = labels_dmap_data_format
        self.return_trafos = return_trafos
        self.transforms = transforms
        self.label_suffix=label_suffix
        self.label_prefix=label_prefix
        self.instance_segmentation_head=instance_segmentation_head

        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        self.samples = os.path.join(self.root_dir,self.samples_dir)
        self.labels  = os.path.join(self.root_dir,self.labels_dir)
        self.labels_dmap  = os.path.join(self.root_dir,self.labels_dmap_dir)

        # Get all sample names sorted as integer values
        all_samples_sorted = sorted(
            glob.glob(f"{self.samples}{os.path.sep}*.{samples_data_format}"),
            key=lambda x: 
                x.split(f"{self.samples}{os.path.sep}")[1].split(
                    f".{samples_data_format}"
            ),
        )
        self.indices = []
        if not empty_dataset:
            # Extract indices from the sorted samples
            self.indices = [
                i.split(f"{self.samples}{os.path.sep}")[1].split(f".{samples_data_format}")[0]
                for i in all_samples_sorted
            ]
        self.raw_mode = False
        self.label_raw_mode = False


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # load sample
        sample_path = os.path.join(self.samples, f"{self.indices[idx]}.{self.samples_data_format}")
        sample_img = tifffile.imread(sample_path) if self.samples_data_format=="tif" else cv2.imread(sample_path,-1)

        if not self.samples_data_format=="tif":
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        
        if sample_img.ndim==2:
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB)
    
        sample_img_lst = []
        label_lst = []
        trafo_lst = []

        if self.labels_available:
            if self.label_raw_mode:
                label_path = os.path.join(self.labels, f"{self.label_prefix}{self.indices[idx]}{self.label_suffix}.{self.labels_data_format}")
                label_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
            else:
                label_path = os.path.join(self.labels_dmap, f"{self.label_prefix}{self.indices[idx]}{self.label_suffix}.{self.labels_dmap_data_format}")
                label_img = tifffile.imread(label_path) if self.labels_dmap_data_format=="tif" else cv2.imread(label_path,-1)
            label_img = label_img.squeeze()
            label_one_hot = np.zeros((label_img.shape[0],label_img.shape[1],1), dtype=np.float32)
            label_one_hot[:,:,0] = label_img.squeeze()
        else:
            label_one_hot = np.zeros((sample_img.shape))

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label_one_hot
            else:
                return sample_img
            
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, label_one_hot)
            sample_img_lst.append(im)
            label_lst.append(lbl)
            trafo_lst.append(trafo)

        if self.instance_segmentation_head:
            sample_img_lst[-1] =  torch.Tensor(slice_image(sample_img_lst[-1].permute(1,2,0).numpy())).permute(0,3,1,2)
            sample_img_lst[-2] = torch.Tensor(slice_image(sample_img_lst[-2].permute(1,2,0).numpy())).permute(0,3,1,2)
        else:
            if len(sample_img_lst) == 4:
                del sample_img_lst[-1]
                del sample_img_lst[-2]


        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            label_lst = label_lst[0] if len(label_lst) > 0 else label_lst
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
       
        # sample_img_lst (optional: labels) (optional: trafos)
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if self.return_trafos and not self.labels_available:
            return sample_img_lst, trafo_lst
        if not self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst, trafo_lst

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        self.indices.append(new_sample)

    def get_samples(self):
        return self.indices