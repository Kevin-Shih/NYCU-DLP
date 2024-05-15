import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, down_sample):
        super().__init__()
        mid_ch = out_ch // 4
        stride = 2 if down_sample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size= 1, bias = False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size= 3, stride= stride, padding= 1, bias = False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(True),
            nn.Conv2d(mid_ch, out_ch, kernel_size= 1, bias = False),
            nn.BatchNorm2d(out_ch)
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

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 7, stride= 2, padding= 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
        )
        self._add_stage(  64,  256, down_sample= False, blocks= 3)
        self._add_stage( 256,  512, down_sample= True,  blocks= 4)
        self._add_stage( 512, 1024, down_sample= True,  blocks= 6)
        self._add_stage(1024, 2048, down_sample= True,  blocks= 3)
        self.fc = nn.Sequential(
            nn.AvgPool2d(kernel_size= 7),
            nn.Flatten(),
            nn.Linear(2048, 100),
        )

    def _add_stage(self, in_ch, out_ch, down_sample, blocks= 3):
        self.features.append(Bottleneck(in_ch, out_ch, down_sample))
        for _ in range(1, blocks):
            self.features.append(Bottleneck(out_ch, out_ch, False))

    def forward(self, x: TensorDataset) -> Tensor:
        features = self.features(x)
        output = self.fc(features)
        return output
