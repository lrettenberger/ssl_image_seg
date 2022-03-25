"""
    Datasets to be used must be specified here to be loadable.
"""
from .base_classes.instance_segmentation.base_inst_seg_data_module import BaseInstanceSegmentationDataModule
from .isic_dermo.isic_dermo_datamodule import IsicDermoDataModule
from .base_classes.segmentation.base_seg_data_module import BaseSegmentationDataModule