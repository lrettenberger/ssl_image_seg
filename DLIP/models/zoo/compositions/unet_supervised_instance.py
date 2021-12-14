from typing import List
import torch
import torch.nn as nn

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder


class UnetInstSegSupervised(BaseComposition):
    
    def __init__(
        self,
        input_channels: int,
        loss_fcn: nn.Module,
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        bilinear = False
        self.append(UnetEncoder(
            input_channels = input_channels,
            encoder_filters = encoder_filters,
            dropout=dropout,
            bilinear=bilinear
        ))
        self.append(UnetDecoder(
            n_classes = 1,
            encoder_filters = encoder_filters,
            decoder_filters = decoder_filters,
            dropout=dropout,
            billinear_downsampling_used = bilinear
        ))
 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
