import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 down: bool,
                 activation_fn: str,
                 use_dropout: bool):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=4, stride=2,
                      padding=1, bias=False,
                      padding_mode='reflect') if down
            else nn.ConvTranspose2d(in_ch, out_ch,
                                    4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU() if activation_fn == 'relu' else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch: int, features: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1 = ConvBlock(features,
                               features * 2,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.down2 = ConvBlock(features * 2,
                               features * 4,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.down3 = ConvBlock(features * 4,
                               features * 8,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.down4 = ConvBlock(features * 8,
                               features * 8,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.down5 = ConvBlock(features * 8,
                               features * 8,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.down6 = ConvBlock(features * 8,
                               features * 8,
                               down=True,
                               activation_fn='leaky',
                               use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8,
                      features * 8,
                      4, 2, 1),
            nn.ReLU()
        )
        self.up1 = ConvBlock(features * 8,
                             features * 8,
                             down=False,
                             activation_fn='relu',
                             use_dropout=True)
        self.up2 = ConvBlock(features * 8 * 2,
                             features * 8,
                             down=False,
                             activation_fn='relu',
                             use_dropout=True)
        self.up3 = ConvBlock(features * 8 * 2,
                             features * 8,
                             down=False,
                             activation_fn='relu',
                             use_dropout=True)
        self.up4 = ConvBlock(features * 8 * 2,
                             features * 8,
                             down=False,
                             activation_fn='relu',
                             use_dropout=False)
        self.up5 = ConvBlock(features * 8 * 2,
                             features * 4,
                             down=False,
                             activation_fn='relu',
                             use_dropout=False)
        self.up6 = ConvBlock(features * 4 * 2,
                             features * 2,
                             down=False,
                             activation_fn='relu',
                             use_dropout=False)
        self.up7 = ConvBlock(features * 2 * 2,
                             features,
                             down=False,
                             activation_fn='relu',
                             use_dropout=False)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(features * 2,
                               in_ch,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x):
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
        return self.output_layer(torch.cat([up7, d0], 1))
