import torch
import torch.nn as nn

from ..modules.block import Bottleneck, C3k, C3k2
from ..modules.conv import Conv
from .dynamic_snake_conv import DySnakeConv


class Bottleneck_DySnakeConv(Bottleneck):
    """Bottleneck with Dynamic Snake Convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = DySnakeConv(c_, c2, k[1])
        self.cv3 = Conv(c2 * 3, c2, k=1)

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C3k_DySnakeConv(C3k):
    """C3k with Dynamic Snake Convolution."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DySnakeConv(C3k2):
    """C3k2 with Dynamic Snake Convolution."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DySnakeConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DySnakeConv(self.c, self.c, shortcut, g) for _ in range(n))


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = torch.roll(x1, self.shift_size, dims=2)
        x2 = torch.roll(x2, -self.shift_size, dims=2)
        x3 = torch.roll(x3, self.shift_size, dims=3)
        x4 = torch.roll(x4, -self.shift_size, dims=3)
        return torch.cat([x1, x2, x3, x4], 1)


class EUCB_SC(nn.Module):
    """Enhanced Upsampling Convolution Block with Shift-Channel mix."""

    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(EUCB_SC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride))
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=True))
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return self.shift_channel_mix(x)
