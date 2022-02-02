from typing import List
import torch
import wandb
import torch.nn as nn

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder


class UnetInstSegSupervised(BaseComposition):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        loss_fcn: nn.Module,
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        dropout: float = 0.0,
        ae_mode = False,
        **kwargs,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.ae_mode = ae_mode
        bilinear = False
        self.append(UnetEncoder(
            input_channels = in_channels,
            encoder_filters = encoder_filters,
            dropout=dropout,
            bilinear=bilinear
        ))
        self.append(UnetDecoder(
            n_classes = out_channels,
            encoder_filters = encoder_filters,
            decoder_filters = decoder_filters,
            dropout=dropout,
            billinear_downsampling_used = bilinear,
            ae_mode = ae_mode
        ))
 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self.log_imgs(x,y_pred)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def log_imgs(self,x,y):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        wandb.log({
            "x": x_wandb,
            "y": y_wandb
        })