import os
import random
import tifffile
import cv2
import numpy as np

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.base_classes.instance_segmentation.base_inst_seg_dataset import BaseInstanceSegmentationDataset


class BaseInstanceSegmentationDataModule(BasePLDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size = 1,
        dataset_size = 1.0,
        val_to_train_ratio = 0,
        initial_labeled_ratio= 1.0,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        samples_dir: str = "samples",
        labels_dir: str = "labels",
        labels_dmap_dir: str = "labels_dist_map",
        label_suffix: str = ''
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size = batch_size,
            val_to_train_ratio = val_to_train_ratio,
            num_workers = num_workers,
            pin_memory = pin_memory,
            shuffle = shuffle,
            drop_last = drop_last,
            initial_labeled_ratio = initial_labeled_ratio,
        )
        if self.initial_labeled_ratio>=0:
            simulated_dataset = True
        else:
            simulated_dataset = False

        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.labels_dir = labels_dir
        self.labels_dmap_dir = labels_dmap_dir
        self.label_suffix = label_suffix

        self.train_labeled_root_dir     = os.path.join(self.root_dir, "train")
        if simulated_dataset:
            self.train_unlabeled_root_dir   = os.path.join(self.root_dir, "train")
        else:
            self.train_unlabeled_root_dir   = os.path.join(self.root_dir,  "unlabeled")
        self.test_labeled_root_dir      = os.path.join(self.root_dir, "test")
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: BaseInstanceSegmentationDataset = None
        self.unlabeled_train_dataset: BaseInstanceSegmentationDataset = None
        self.val_dataset: BaseInstanceSegmentationDataset = None
        self.test_dataset: BaseInstanceSegmentationDataset = None
        self.samples_data_format, self.labels_data_format, self.labels_dmap_data_format = self._determine_data_format()
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = BaseInstanceSegmentationDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.train_transforms,
            samples_dir=self.samples_dir,
            labels_dir=self.labels_dir,
            labels_dmap_dir=self.labels_dmap_dir,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            labels_dmap_data_format=self.labels_dmap_data_format,
            label_suffix=self.label_suffix
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.val_dataset = BaseInstanceSegmentationDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.val_transforms,
            empty_dataset=True,
            samples_dir=self.samples_dir,
            labels_dir=self.labels_dir,
            labels_dmap_dir=self.labels_dmap_dir,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            labels_dmap_data_format=self.labels_dmap_data_format,
            label_suffix=self.label_suffix
        )

        self.unlabeled_train_dataset = BaseInstanceSegmentationDataset(
            root_dir=self.train_unlabeled_root_dir,
            transforms=self.train_transforms,
            labels_available=False,
            samples_dir=self.samples_dir,
            labels_dir=self.labels_dir,
            labels_dmap_dir=self.labels_dmap_dir,
            return_trafos=self.return_unlabeled_trafos,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            labels_dmap_data_format=self.labels_dmap_data_format,
            label_suffix=self.label_suffix
        )
        
        self.test_dataset = BaseInstanceSegmentationDataset(
            root_dir=self.test_labeled_root_dir, 
            transforms=self.test_transforms,
            samples_dir=self.samples_dir,
            labels_dir=self.labels_dir,
            labels_dmap_dir=self.labels_dmap_dir,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            labels_dmap_data_format=self.labels_dmap_data_format,
            label_suffix=self.label_suffix
        )

    def _determine_data_format(self):
        extensions = {self.samples_dir: list(), self.labels_dir: list(), self.labels_dmap_dir: list()}

        for folder in extensions.keys():
            for file in os.listdir(os.path.join(self.train_labeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

            for file in os.listdir(os.path.join(self.train_unlabeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

        return max(set(extensions[self.samples_dir]), key = extensions[self.samples_dir].count),\
               max(set(extensions[self.labels_dir]), key = extensions[self.labels_dir].count), \
               max(set(extensions[self.labels_dmap_dir]), key = extensions[self.labels_dmap_dir].count) 