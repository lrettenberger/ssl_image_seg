# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import matplotlib.pyplot as plt
from math import ceil,floor

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
            self,
            patch_size=32,
            num_classes=1000,
            in_chans=1, 
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768], 
            drop_path_rate=0., 
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(patch_size//8), stride=(patch_size//8)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # muss man wohl wieder loeschen. avgpool braucht man weils DenseCl erwartet weil eigentlich mit Resnet50 gemacht wird.
        self.fc = nn.Linear(dims[-1], num_classes)
        self.avgpool = nn.Identity()


    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            #self.plot_features(x,name=i)
        # disable norm for densecl ssl training
        return x
        #return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x
    
    def plot_features(self,x,batch_idx=0,name='feature_plot'):
        x_idx = x[batch_idx]
        num_feats,h,w = x_idx.shape
        fig, axs = plt.subplots(ceil(num_feats ** .5), floor(num_feats ** .5),gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        indx=0
        for x in range(ceil(num_feats ** .5)):
            for y in range(floor(num_feats ** .5)):
                if indx < len(x_idx):
                    axs[x,y].imshow(x_idx[indx].detach().cpu().numpy(), aspect='auto')
                axs[x,y].set_axis_off()
                indx+=1
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(f'{name}.png')
        plt.close()

class ConvNeXtV2Atto(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[2, 2, 6, 2], 
            dims=[40, 80, 160, 320],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)

class ConvNeXtV2Femto(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[2, 2, 6, 2], 
            dims=[48, 96, 192, 384],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)

class ConvNeXtV2Pico(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[2, 2, 6, 2], 
            dims=[64, 128, 256, 512],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)

class ConvNeXtV2Nano(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[2, 2, 8, 2], 
            dims=[80, 160, 320, 640],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)


class ConvNeXtV2Tiny(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)

class ConvNeXtV2Base(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[3, 3, 27, 3], 
            dims=[128, 256, 512, 1024],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)


class ConvNeXtV2Large(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[3, 3, 27, 3], 
            dims=[192, 384, 768, 1536],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)

class ConvNeXtV2Huge(ConvNeXtV2):
    def __init__(
            self, 
            patch_size=32, 
            num_classes=1000, 
            in_chans=1, 
            depths=[3, 3, 27, 3], 
            dims=[352, 704, 1408, 2816],
            drop_path_rate=0
        ):
        super().__init__(patch_size, num_classes, in_chans, depths, dims, drop_path_rate)
    


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model