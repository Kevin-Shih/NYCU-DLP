# this code is based on https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py, which is released under the MIT licesne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

""" class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        #     # nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        # )

    def forward(self, x):
        # x1 = x.detach().clone()
        # x2 = self.up(x)
        return self.relu(self.up(x) + self.shortcut(x))
 """
class GenBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
        )
    def forward(self, x):
        return self.up(x)
""" 
class GenBlock4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
        )
    def forward(self, x):
        return self.up(x)
 """
class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed, self).__init__()
        # self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x).view(-1, self.emb_dim, 1, 1)

""" class Embed2(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed2, self).__init__()
        # self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x).view(-1, self.emb_dim, 1, 1) """
""" 
class Generator(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class=24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding = Embed(num_class, cond_dim)
        self.up1 = nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, bias=False)
        self.G1 = GenBlock(z_dim*8, z_dim*4)
        self.G2 = GenBlock(z_dim*4, z_dim*2)
        self.G3 = GenBlock(z_dim*2, z_dim)
        # self.G4 = GenBlock(z_dim*2, z_dim)
        self.up2 = nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up1.weight)
        init.xavier_uniform_(self.up2.weight)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)

    def forward(self, x, y):
        x = x.view(-1, self.z_dim, 1, 1)
        y = self.embedding(y)
        input = self.up1(torch.cat((x, y), 1))

        input = self.G1(input)
        input = self.G2(input)
        input = self.G3(input)
        out = self.up2(input)
        out = nn.Tanh()(out)
        return out

class Generator3(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class=24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding = Embed2(num_class, cond_dim)
        self.up1 = nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, 0, bias=False)
        self.G1 = GenBlock3(z_dim*8, z_dim*4)
        self.G2 = GenBlock3(z_dim*4, z_dim*2)
        self.G3 = GenBlock3(z_dim*2, z_dim)
        # self.G4 = GenBlock(z_dim*2, z_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up1.weight)
        # init.xavier_uniform_(self.up2.weight)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)

    def forward(self, x, y):
        x = x.view(-1, self.z_dim, 1, 1)
        y = self.embedding(y)
        input = self.up1(torch.cat((x, y), 1))

        input = self.G1(input)
        input = self.G2(input)
        input = self.G3(input)
        out = self.up2(input)
        return out

class Generator3a(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class=24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding = Embed2(num_class, cond_dim)
        self.up1 = nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, 0, bias=False)
        self.G1 = GenBlock4(z_dim*8, z_dim*4)
        self.G2 = GenBlock4(z_dim*4, z_dim*2)
        self.G3 = GenBlock4(z_dim*2, z_dim)
        # self.G4 = GenBlock(z_dim*2, z_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up1.weight)
        # init.xavier_uniform_(self.up2.weight)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)

    def forward(self, x, y):
        x = x.view(-1, self.z_dim, 1, 1)
        y = self.embedding(y)
        input = self.up1(torch.cat((x, y), 1))

        input = self.G1(input)
        input = self.G2(input)
        input = self.G3(input)
        out = self.up2(input)
        return out
 """
class Generator4(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class=24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding0 = Embed(num_class, cond_dim)
        self.embedding1 = Embed(num_class, z_dim*8)
        self.embedding2 = Embed(num_class, z_dim*4)
        self.embedding3 = Embed(num_class, z_dim*2)
        self.up1 = nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, 0, bias=False)
        self.G1 = GenBlock3(z_dim*8, z_dim*4)
        self.G2 = GenBlock3(z_dim*4, z_dim*2)
        self.G3 = GenBlock3(z_dim*2, z_dim)
        # self.G4 = GenBlock(z_dim*2, z_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up1.weight)
        # init.xavier_uniform_(self.up2.weight)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)

    def forward(self, x, y):
        x = x.view(-1, self.z_dim, 1, 1)
        y0  = self.embedding0(y)
        y1 = self.embedding1(y)
        y2 = self.embedding2(y)
        y3 = self.embedding3(y)
        input = self.up1(torch.cat((x, y0), 1))

        input = self.G1(y1*input)
        input = self.G2(y2*input)
        input = self.G3(y3*input)
        out = self.up2(input)
        return out

"""
class Generator4a(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class=24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding0 = Embed(num_class, cond_dim)
        self.embedding1 = Embed(num_class, z_dim*8)
        self.embedding2 = Embed(num_class, z_dim*4)
        self.embedding3 = Embed(num_class, z_dim*2)
        self.up1 = nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, 0, bias=False)
        self.G1 = GenBlock4(z_dim*8, z_dim*4)
        self.G2 = GenBlock4(z_dim*4, z_dim*2)
        self.G3 = GenBlock4(z_dim*2, z_dim)
        # self.G4 = GenBlock(z_dim*2, z_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up1.weight)
        # init.xavier_uniform_(self.up2.weight)
        for m in self.G1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        for m in self.G3.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)

    def forward(self, x, y):
        x = x.view(-1, self.z_dim, 1, 1)
        y0  = self.embedding0(y)
        y1 = self.embedding1(y)
        y2 = self.embedding2(y)
        y3 = self.embedding3(y)
        input = self.up1(torch.cat((x, y0), 1))

        input = self.G1(y1*input)
        input = self.G2(y2*input)
        input = self.G3(y3*input)
        out = self.up2(input)
        return out

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dsample, num):
        super().__init__()
        size = 64 // (2 ** (num - 1))
        stride = 2 if dsample else 1
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.LayerNorm([in_channels, size, size], elementwise_affine= False),# True
            # nn.GroupNorm(4, in_channels, affine= False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.LayerNorm([out_channels, size//stride, size//stride], elementwise_affine= False),# True
            # nn.GroupNorm(8, out_channels, affine= False),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.initialize()

    def initialize(self):
        for m in self.res.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, np.sqrt(2))
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.relu(self.res(x) + self.shortcut(x))

class DiscBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, dsample, num):
        super().__init__()
        size = 64 // (2 ** (num - 1))
        stride = 2 if dsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, 1, 1, bias=False),
            nn.LayerNorm([in_channels*2, size, size], elementwise_affine= False),# True
            # nn.GroupNorm(8, in_channels*2),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels*2, out_channels, 3, stride, 1, bias=False),
            nn.LayerNorm([out_channels, size//stride, size//stride], elementwise_affine= False),# True
            # nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True)
            # nn.GELU(),
        )
        self.initialize()

    def initialize(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, np.sqrt(2))

    def forward(self, x):
        return self.conv(x)

 class Discriminator(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, z_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            DiscBlock(  z_dim, z_dim*2,  True, num=2),
            nn.Dropout(0.5),
            DiscBlock(z_dim*2, z_dim*4,  True, num=3),
            nn.Dropout(0.5),
            DiscBlock(z_dim*4, z_dim*8, False, num=4),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(z_dim*8, 1)
        self.ac = nn.Sequential(nn.Linear(z_dim*8, num_class), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out
    
class Discriminator3(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, z_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),

            DiscBlock3(  z_dim, z_dim*2,  True, num= 2),
            nn.Dropout(0.5),
            DiscBlock3(z_dim*2, z_dim*8,  True, num= 3),
            nn.Dropout(0.5),
            nn.Conv2d(z_dim*8, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*16, 4, 4], elementwise_affine= False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(z_dim*16, 1)
        self.ac = nn.Sequential(nn.Linear(z_dim*16, num_class), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out """

class Discriminator4(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, z_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(z_dim, z_dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([z_dim*2, 32, 32], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(z_dim*2, z_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*4, 16, 16], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(z_dim*4, z_dim*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([z_dim*8, 16, 16], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*8, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*16, 8, 8], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(z_dim*16, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*16, 4, 4], elementwise_affine= False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.linear = nn.Linear(z_dim*16*4*4, 1)
        self.ac = nn.Sequential(nn.Linear(z_dim*16*4*4, num_class), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out
    
""" class Discriminator4a(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, z_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.5),

            nn.Conv2d(z_dim, z_dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([z_dim*2, 32, 32], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*2, z_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*4, 16, 16], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*4, z_dim*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([z_dim*8, 16, 16], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*8, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([z_dim*16, 8, 8], elementwise_affine= False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*16, z_dim*16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([z_dim*16, 6, 6], elementwise_affine= False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.linear = nn.Linear(z_dim*16*36, 1)
        self.ac = nn.Sequential(nn.Linear(z_dim*16*36, num_class), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out

class Discriminator4b(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, z_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.5),

            nn.Conv2d(z_dim, z_dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(z_dim*2),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.5),

            nn.Conv2d(z_dim*2, z_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(z_dim*4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*4, z_dim*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(z_dim*8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*8, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(z_dim*16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(z_dim*16, z_dim*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(z_dim*16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.linear = nn.Linear(z_dim*16*4*4, 1)
        self.ac = nn.Sequential(nn.Linear(z_dim*16*4*4, num_class), nn.Softmax(dim=1))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        wgan_out = self.linear(x)
        ac_out = self.ac(x)
        return wgan_out, ac_out """