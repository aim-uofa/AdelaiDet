# coding:utf-8
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same',
                 stride=1, dilation=1, groups=1):
        super(Conv2D, self).__init__()

        assert type(kernel_size) in [int, tuple], "Allowed kernel type [int or tuple], not {}".format(type(kernel_size))
        assert padding == 'same', "Allowed padding type {}, not {}".format('same', padding)

        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            self.h_kernel = kernel_size[0]
            self.w_kernel = kernel_size[1]
        else:
            self.h_kernel = kernel_size
            self.w_kernel = kernel_size

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=self.stride, dilation=self.dilation, groups=self.groups)

    def forward(self, x):

        if self.padding == 'same':

            height, width = x.shape[2:]

            h_pad_need = max(0, (height - 1) * self.stride + self.h_kernel - height)
            w_pad_need = max(0, (width - 1) * self.stride + self.w_kernel - width)

            pad_left = w_pad_need // 2
            pad_right = w_pad_need - pad_left
            pad_top = h_pad_need // 2
            pad_bottom = h_pad_need - pad_top

            padding = (pad_left, pad_right, pad_top, pad_bottom)

            x = F.pad(x, padding, 'constant', 0)

        x = self.conv(x)

        return x


class GCN(nn.Module):
    """
        Large Kernel Matters -- https://arxiv.org/abs/1703.02719
    """
    def __init__(self, in_channels, out_channels, k=3):
        super(GCN, self).__init__()

        self.conv_l1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')
        self.conv_l2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')

        self.conv_r1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')
        self.conv_r2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)

        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)

        out = x1 + x2

        return out
