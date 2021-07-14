from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resmap = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = self.resmap(x)

        return x + residual

class convlayer(nn.Module):
    def __init__(self, nIn, nOut, k = 3, p = 1, s = 1, d = 1):
        super(convlayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=k, padding=p, stride=s, dilation=d),
            nn.BatchNorm2d(nOut),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class deconvlayer(nn.Module):
    def __init__(self, nIn, nOut, k = 4, p = 1, s = 2):
        super(deconvlayer, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(nIn, nOut, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(nOut),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class GlobalNet(nn.Module):
    def __init__(self, channels_input):
        super(GlobalNet, self).__init__()
        filters = [128, 256, 512, 1024]

        self.conv1 = convlayer(channels_input, filters[0])
        self.res1 = ResidualBlock(filters[0])
        self.conv2 = convlayer(filters[0], filters[1], k = 2, p = 0, s = 2)
        self.res2 = ResidualBlock(filters[1])
        self.conv3 = convlayer(filters[1], filters[2], k = 2, p = 0, s = 2)
        self.conv3a = convlayer(filters[2], filters[2])
        self.conv3b = convlayer(filters[2], filters[2])
        self.conv3c = convlayer(filters[2], filters[2])

        self.deconv1 = deconvlayer(filters[2], filters[1])
        self.deconv2 = deconvlayer(filters[1], filters[0])

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        res1 = self.res1(conv1)
        conv2 = self.conv2(res1)
        res2 = self.res2(conv2)
        conv3 = self.conv3(res2)
        conv3 = self.conv3a(conv3)
        conv3 = self.conv3b(conv3)
        conv3 = self.conv3c(conv3)

        deconv1 = self.deconv1(conv3)
        elt01 = deconv1 + res2

        deconv2 = self.deconv2(elt01)
        elt02 = deconv2 + res1

        return elt02

class Mid_Net(nn.Module):
    def __init__(self, channels_input):
        super(Mid_Net, self).__init__()
        filters = [128, 256, 512, 1024]
        self.conv1 = convlayer(channels_input, filters[2], k = 4, p = 0, s = 4)
        self.conv2 = convlayer(filters[2], filters[2])
        self.conv3 = convlayer(filters[2], filters[3], k = 2, p = 0, s = 2)
        self.conv4 = convlayer(filters[3], filters[3])

        self.deconv1 = deconvlayer(filters[3], filters[2])
        self.deconv2 = deconvlayer(filters[2], filters[1])
        self.deconv3 = deconvlayer(filters[1], filters[0])

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        deconv1 = self.deconv1(conv4)
        elt01 = deconv1 + conv2

        deconv2 = self.deconv2(elt01)

        deconv3 = self.deconv3(deconv2)

        return deconv3

class LocalNet(nn.Module):
    def __init__(self, channels_input):
        super(LocalNet, self).__init__()
        filters = [128, 256, 512, 1024]

        self.conv1 = convlayer(channels_input, filters[3], k = 8, p = 0, s = 8)
        self.conv2 = convlayer(filters[3], filters[2], k = 1, p = 0, s = 1)
        self.conv3 = convlayer(filters[2], filters[2])
        self.conv4 = convlayer(filters[2], filters[2])
        self.conv5 = convlayer(filters[2], filters[3], k = 1, p = 0, s = 1)

        self.deconv1 = deconvlayer(filters[3], filters[2])
        self.deconv2 = deconvlayer(filters[2], filters[1])
        self.deconv3 = deconvlayer(filters[1], filters[0])

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv1 = self.deconv1(conv5)
        deconv2 = self.deconv2(deconv1)
        deconv3 = self.deconv3(deconv2)

        return deconv3

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class Origin_Generator_Radv2(nn.Module):
    def __init__(self, channels_input, channels_output):
        super(Origin_Generator_Radv2, self).__init__()

        self.Global = GlobalNet(channels_input)
        self.Medium = Mid_Net(channels_input)
        self.Local = LocalNet(channels_input)

        ##### Fuse net
        self.fuse = nn.Conv2d(128 * 3, channels_output, kernel_size = 3, padding = 1, stride = 1)

    def forward(self, x):

        Global = self.Global(x)
        Medium = self.Medium(x)
        Local = self.Local(x)

        ##### Fuse
        result = self.fuse(torch.cat((Global, Medium, Local), dim = 1))

        return result

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
