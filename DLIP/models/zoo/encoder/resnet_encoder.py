from typing import List
import torch.nn as nn
import logging
import numpy as np
import os
import torchvision
import torch

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv
from DLIP.models.zoo.building_blocks.down_sample import Down
from DLIP.models.zoo.encoder.basic_encoder import BasicEncoder

class ResNetEncoder(BasicEncoder):
    def __init__(
        self,
        input_channels: int,
        encoder_type = 'resnet50',
        pretraining_weights = 'imagenet',
        encoder_frozen=False
    ):
        super().__init__(input_channels)
        encoder_class = None
        encoder_type = encoder_type.lower()
        # Its a resnet encoder!
        if encoder_type == 'resnet18':
            encoder_class = torchvision.models.resnet18
        if encoder_type == 'resnet34':
            encoder_class = torchvision.models.resnet34
        if encoder_type == 'resnet50':
            encoder_class = torchvision.models.resnet50
        if encoder_type == 'resnet101':
            encoder_class = torchvision.models.resnet101
        if encoder_type == 'resnet152':
            encoder_class = torchvision.models.resnet152
        if encoder_class != None:
            # we determined resnet encoder type. Now we put it into the backbone module list
            if pretraining_weights == 'imagenet':
                logging.info('Loading imagenet weights')
            encoder = encoder_class(pretrained=True if pretraining_weights == 'imagenet' else False)
            if input_channels != 3:
                encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=encoder.conv1.kernel_size, stride=encoder.conv1.stride, padding=encoder.conv1.padding,bias=encoder.conv1.bias)
            # Load pretraining weights
            if pretraining_weights != 'imagenet' and pretraining_weights != None:
                logging.info(f'Loading weights ({pretraining_weights}) ...')
                if not os.path.exists(pretraining_weights):
                    raise ValueError(f'Pretraining weights path does not exists ({pretraining_weights})')
                if os.path.isdir(pretraining_weights):
                    logging.info('It guess the filename (dnn_weights.ckpt) was forgotten')
                    pretraining_weights = os.path.join(pretraining_weights,'dnn_weights.ckpt')
                weights = torch.load(pretraining_weights)
                weights = weights['state_dict']
                for key in list(weights.keys()):
                    weights[key.replace('encoder.','')] = weights[key]
                    del weights[key]
                encoder.load_state_dict(weights)
                logging.info(f'Loaded weights.')
            if encoder_frozen:
                for param in encoder.parameters():
                            param.requires_grad = False
            encoder_layers = list(encoder.children())
            self.backbone.append(nn.Sequential(*encoder_layers[:3]))
            self.backbone.append(nn.Sequential(*encoder_layers[3:5]))
            self.backbone.append(encoder_layers[5])
            self.backbone.append(encoder_layers[6])
            self.backbone.append(encoder_layers[7])
        if encoder_class == None:
            raise ValueError(f'Could not find encoder {encoder_type}!')
