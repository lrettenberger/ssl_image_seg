"""
    Datasets to be used must be specified here to be loadable.
"""
from .base_classes.instance_segmentation.base_inst_seg_data_module import BaseInstanceSegmentationDataModule
from .isic_dermo.isic_dermo_datamodule import IsicDermoDataModule
from .base_classes.segmentation.base_seg_data_module import BaseSegmentationDataModule
from .cats_vs_dogs.cats_vs_dogs_datamodule import CatsVsDogsDatamodule
from .bear_detection.bear_detection_datamodule import BearDetectionDataModule
from .coco.coco_datamodule import CocoDataModule
from .nih_chest_xray.NIH_chest_xray_datamodule import NIHChestXrayDataModule
from .pneumonia_xray.pneumonia_xray_datamodule import PneumoniaXrayDataModule