import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset

def _add_conv_block(conv_block:nn.Sequential, in_ch, out_ch, layers= 2):
    for i in range(layers):
        conv_block.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                                    kernel_size= 3, padding= 1, bias= False))
        conv_block.append(nn.BatchNorm2d(out_ch))
        conv_block.append(nn.ReLU(True))
    conv_block.append(nn.MaxPool2d(2, 2))
    return conv_block

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.features = _add_conv_block(self.features, 3, 64, 2)
        self.features = _add_conv_block(self.features, 64, 128, 2)
        self.features = _add_conv_block(self.features, 128, 256, 4)
        self.features = _add_conv_block(self.features, 256, 512, 4)
        self.features = _add_conv_block(self.features, 512, 512, 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 100)
        )

    def forward(self, x: TensorDataset) -> Tensor:
        features = self.features(x)
        output = self.fc(features)
        return output