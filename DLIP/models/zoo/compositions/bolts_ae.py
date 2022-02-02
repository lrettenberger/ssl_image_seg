from pl_bolts.models.autoencoders import AE
from pl_bolts.models.autoencoders import VAE
import torch.nn as nn
import torch
import wandb
import pytorch_lightning as pl

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv

class ConvReluDropout(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout=0.0,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class BoltsAEDecoder(nn.Module):   
    
    def forward(self, z):
        z = z.unsqueeze(dim=2).unsqueeze(dim=2)
        x = self.up_conv6(z)
        x = self.conv6(x)
        x = self.up_conv7(x)
        x = self.conv7(x)
        x = self.up_conv8(x)
        x = self.conv8(x)
        x = self.up_conv9(x)
        x = self.conv9(x)
        x = self.up_conv10(x)
        x = self.conv10(x)
        return self.sigmoid(x)
    
    
    def __init__(self,latent_dim,out_channels):
        super().__init__()
        self.up_conv6 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=2)
        self.conv6 = ConvReluDropout(512, 512)
        self.up_conv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvReluDropout(256, 256)
        self.up_conv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvReluDropout(128, 128)
        self.up_conv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvReluDropout(64, 64)
        self.up_conv10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1) # outconv
        self.sigmoid = nn.Sigmoid()

class BoltsAE(AE, pl.LightningModule):

    def __init__(self, input_height: int,  loss_fcn: nn.Module, in_channels: int = 3, enc_type: str = 'resnet18', first_conv: bool = True, maxpool1: bool = True, enc_out_dim: int = 512, latent_dim: int = 2048, lr: float = 0.001,out_channels=3,ae_mode=False, **kwargs):
        super(BoltsAE,self).__init__(input_height=input_height, enc_type=enc_type, first_conv=first_conv, maxpool1=maxpool1, enc_out_dim=enc_out_dim, latent_dim=latent_dim, lr=lr, **kwargs)
        #self.fc = nn.Identity()
        self.decoder.conv1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.decoder = BoltsAEDecoder(latent_dim,out_channels)
        self.loss_fcn = loss_fcn
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()


    def set_optimizers(self, optimizer, lrs=None, metric_to_track=None):
        self.optimizer = optimizer
        self.lrs = lrs
        self.metric_to_track = metric_to_track
        if self.metric_to_track  is None:
            self.metric_to_track = "val/loss"

    def configure_optimizers(self):
        if self.lrs is None and self.metric_to_track is None:
            return {"optimizer": self.optimizer}
        if self.lrs is None:
            return {"optimizer": self.optimizer, "monitor": self.metric_to_track}
        if self.metric_to_track is None:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lrs}
        return {"optimizer": self.optimizer,"lr_scheduler": self.lrs,"monitor": self.metric_to_track}

    def get_progress_bar_dict(self):
        # don't show the running loss (very iritating)
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    def training_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        recon_loss = torch.mean(recon_loss)
        self.log("train/loss", recon_loss, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        loss = torch.mean(recon_loss)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self.log_imgs(x,x_hat)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        loss = torch.mean(recon_loss)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def log_imgs(self,x,y,img_limit=3):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        wandb.log({
            "x": x_wandb[:img_limit],
            "y": y_wandb[:img_limit]
        })

