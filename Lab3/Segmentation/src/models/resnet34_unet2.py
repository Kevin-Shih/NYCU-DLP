import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, down_sample):
        super().__init__()
        stride = 2 if down_sample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size= 3, stride= stride, padding= 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride= stride, bias = False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: TensorDataset) -> Tensor:
        output = nn.functional.relu(self.conv(x) + self.shortcut(x), True)
        return output

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

class ResNet34_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 7, stride= 2, padding= 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
        self.stage1 = self._build_stage( 64,  64, False, 3)
        self.stage2 = self._build_stage( 64, 128,  True, 4)
        self.stage3 = self._build_stage(128, 256,  True, 6)
        self.stage4 = self._build_stage(256, 512,  True, 3)
        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size= 2, stride= 2)
        self.dconv1 = ConvBlock(1024, 256)
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size= 2, stride= 2)
        self.dconv2 = ConvBlock(512, 128)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size= 2, stride= 2)
        self.dconv3 = ConvBlock(256, 64)
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size= 2, stride= 2)
        self.dconv4 = ConvBlock(128, 32)
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size= 2, stride= 2)
        self.dconv5 = nn.Conv2d(32, 1, kernel_size= 1)

    def _build_stage(self, in_ch, out_ch, down_sample, blocks= 3):
        stage = nn.Sequential(Basic(in_ch, out_ch, down_sample))
        for _ in range(1, blocks):
            stage.append(Basic(out_ch, out_ch, False))
        return stage

    def forward(self, x: TensorDataset) -> Tensor:
        c0 = self.stage0(x)
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        u1 = self.up1(torch.cat([c5, c4], dim= 1))
        dc1 = self.dconv1(u1)
        u2 = self.up2(torch.cat([dc1, c3], dim= 1))
        dc2 = self.dconv2(u2)
        u3 = self.up3(torch.cat([dc2, c2], dim= 1))
        dc3 = self.dconv3(u3)
        u4 = self.up4(torch.cat([dc3, c1], dim= 1))
        dc4 = self.dconv4(u4)
        u5 = self.up5(dc4)
        output = self.dconv5(u5)
        output = nn.functional.sigmoid(output)
        return output