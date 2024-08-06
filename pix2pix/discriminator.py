import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch: int, features=[64, 128, 256, 512]):
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
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2
            layers.append(CNNBlock(in_ch, feature, stride=stride))
            in_ch = feature
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.input(x)
        x = self.model(x)
        return x
