from typing import List
import torch
import torch.nn as nn

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder


class UnetSupervised(BaseComposition):
    
    def __init__(
        self,
        n_classes: int,
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
            n_classes = n_classes,
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
        self.log_metrics(y_pred, y_true, mode="train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc",torch.sum(torch.round(y_pred) == y_true) / len(torch.flatten(y_true)),prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log_metrics(y_pred, y_true, mode="val")
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc",torch.sum(torch.round(y_pred) == y_true) / len(torch.flatten(y_true)),prog_bar=True,on_epoch=True, on_step=False,)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log_metrics(y_pred, y_true, mode="test")
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def log_metrics(self, y_pred, y_true, mode):
        dice_score = self.dice_score(y_pred, y_true)
        jaccard_score = self.jaccard_score(y_pred, y_true)

        # metrics per class
        for i_c in range(y_pred.shape[1]):
            self.log(
                f"{mode}/jaccard_score_class_{i_c}",
                jaccard_score[i_c],
                on_epoch=True,
                on_step=False,
            )
            self.log(
                f"{mode}/dice_score_class_{i_c}",
                dice_score[i_c],
                on_epoch=True,
                on_step=False,
            )

        # mean metrics
        mean_jaccard_score = torch.mean(jaccard_score)
        mean_dice_score = torch.mean(dice_score)

        self.log(
            f"{mode}/jaccard_score_mean",
            mean_jaccard_score,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"{mode}/dice_score_mean", mean_dice_score, on_epoch=True, on_step=False
        )
