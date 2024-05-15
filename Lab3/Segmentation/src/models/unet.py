import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size= 3, padding= 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size= 3, padding= 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x: TensorDataset) -> Tensor:
        output = self.conv(x)
        return output

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(  3,  64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock( 64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size= 2, stride= 2)
        self.dconv1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size= 2, stride= 2)
        self.dconv2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size= 2, stride= 2)
        self.dconv3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size= 2, stride= 2)
        self.dconv4 = ConvBlock(128, 64)
        self.dconv5 = nn.Conv2d(64, 1, kernel_size= 1)

    def forward(self, x: TensorDataset) -> Tensor:
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        u1 = torch.cat([self.up1(c5), c4], dim= 1)
        dc1 = self.dconv1(u1)
        u2 = torch.cat([self.up2(dc1), c3], dim= 1)
        dc2 = self.dconv2(u2)
        u3 = torch.cat([self.up3(dc2), c2], dim= 1)
        dc3 = self.dconv3(u3)
        u4 = torch.cat([self.up4(dc3), c1], dim= 1)
        dc4 = self.dconv4(u4)
        output = self.dconv5(dc4)
        output = nn.Sigmoid()(output)
        return output