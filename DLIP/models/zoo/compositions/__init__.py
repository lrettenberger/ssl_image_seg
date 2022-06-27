"""
    Models to be used must be specified here to be loadable.
"""
#from .unet_supervised_instance import UnetInstSegSupervised
from .example_classifier import ExampleClassifier
from .bolts_ae import BoltsAE
from .s_resnet_unet import SResUnet
from .smp_unet_resnet import SmpUnetResnet
from .moco_v2 import Mocov2
from .resnet_classifier import ResnetClassifier
from .detectron_instance import Detectron2Instance
from .detectron_semantic import Detectron2Semantic
from .unet_instance import UnetInstance