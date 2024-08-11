"""
PyTorch based Discriminator model used in a conditional GAN.

Classes:
    - ConvBlock: Sequential model with Conv2D, InstanceNorm and LeakyReLU layers.
    - Discriminator: Model used to distinguish between real and generated images.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv2D layer followed by InstanceNorm and LeakyReLU.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        stride (int): Conv2D stride.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Pass data through the block.
        """
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Discriminator model applying a series of convolutional layers to input images to
    classify them as real or generated.

    Args:
        in_ch (int): Number of input channels.
        features (list[int]): List of feature sizes for each layer.
    """
    def __init__(self, in_ch: int, features=[128, 256, 512, 1024]):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_ch * 2,
                      features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_ch = features[0]
        for idx, feature in enumerate(features[1:]):
            layers.append(ConvBlock(in_ch,
                                    feature,
                                    stride=1 if feature == features[-1] else 2))
            in_ch = feature
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        self.model = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling

    def forward(self, x, y):
        """
        Forward pass through the model.
        """
        x = torch.cat([x, y], dim=1)
        x = self.input(x)
        features = self.model(x)
        return self.gap(features).view(x.size(0), -1), features
