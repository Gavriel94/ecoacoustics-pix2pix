"""
Generator model for a conditional GAN.

Classes:
    - DownBlock: Sequential model with Conv2D, BatchNorm2D and LeakyReLU layers.
    - UpBlock: Sequential model with Conv2D, BatchNorm2D and LeakyReLU layers.
    - Discriminator: Model used to distinguish between real and generated images.
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """
    A downsampling block used in the conditional GAN.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        use_dropout (bool, optional): Apply dropout after convolution. Default is False.
    """
    def __init__(self, in_ch: int, out_ch: int, use_dropout: bool = False):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Pass data through the block.
        """
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UpBlock(nn.Module):
    """
    An upsampling block used in the conditional GAN.
    Images are upsampled using regular convolutions.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        use_dropout (bool, optional): Apply dropout after convolution. Default is False.
    """
    def __init__(self, in_ch: int, out_ch: int, use_dropout: bool = False):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Pass data through the block.
        """
        x = self.up(x)
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    Generator model used to synthesise images based on input and target
    images.

    Args:
        in_ch (int): Number of input channels
        features (int): Number of features, defining the depth of the model.
    """
    def __init__(self, in_ch: int, features: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1 = DownBlock(features, features * 2)
        self.down2 = DownBlock(features * 2, features * 4)
        self.down3 = DownBlock(features * 4, features * 8)
        self.down4 = DownBlock(features * 8, features * 8)
        self.down5 = DownBlock(features * 8, features * 8)
        self.down6 = DownBlock(features * 8, features * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )
        self.up1 = UpBlock(features * 8, features * 8, use_dropout=True)
        self.up2 = UpBlock(features * 8 * 2, features * 8, use_dropout=True)
        self.up3 = UpBlock(features * 8 * 2, features * 8, use_dropout=True)
        self.up4 = UpBlock(features * 8 * 2, features * 8)
        self.up5 = UpBlock(features * 8 * 2, features * 4)
        self.up6 = UpBlock(features * 4 * 2, features * 2)
        self.up7 = UpBlock(features * 2 * 2, features)
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features * 2, in_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Pass data through the U-Net model.
        """
        d0 = self.input_layer(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bn = self.bottleneck(d6)
        up1 = self.up1(bn)
        up2 = self.up2(torch.cat([up1, d6], 1))
        up3 = self.up3(torch.cat([up2, d5], 1))
        up4 = self.up4(torch.cat([up3, d4], 1))
        up5 = self.up5(torch.cat([up4, d3], 1))
        up6 = self.up6(torch.cat([up5, d2], 1))
        up7 = self.up7(torch.cat([up6, d1], 1))
        out = self.out(torch.cat([up7, d0], 1))
        return out
