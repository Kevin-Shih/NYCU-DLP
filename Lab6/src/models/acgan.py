# this code is based on https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py, which is released under the MIT licesne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        #     # nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        # )

    def forward(self, x):
        # x1 = x.detach().clone()
        # x2 = self.up(x)
        return self.lrelu(self.up(x) + self.shortcut(x))

class GenBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
            # nn.GELU()
        )
    def forward(self, x):
        return self.up(x)

class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed, self).__init__()
        # self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LeakyReLU(inplace=True)
            # nn.GELU()
        )

    def forward(self, x):
        return self.model(x).view(-1, self.emb_dim, 1, 1)

class Generator(nn.Module):
    def __init__(self, z_dim = 128, cond_dim = 128, num_class = 24):
        super().__init__()
        self.z_dim = z_dim
        self.embedding = Embed(num_class, cond_dim)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim + cond_dim, z_dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(z_dim*8),
            nn.LeakyReLU(inplace=True)
            # nn.GELU()
        )
        self.G1 = GenBlock2(z_dim*8, z_dim*4)
        self.G2 = GenBlock2(z_dim*4, z_dim*2)
        self.G3 = GenBlock2(z_dim*2, z_dim)
        self.up2 = nn.ConvTranspose2d(z_dim, 3, 4, 2, 1, bias=False)
        self.initialize()

    def initialize(self):
        # init.xavier_uniform_(self.up1.weight)
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

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dsample):
        super().__init__()
        stride = 2 if dsample else 1
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.initialize()

    def initialize(self):
        for m in self.res.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, np.sqrt(2))
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.lrelu(self.res(x) + self.shortcut(x))


class DiscBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, dsample):
        super().__init__()
        stride = 2 if dsample else 1
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # self.lrelu = nn.GELU()
        self.initialize()

    def initialize(self):
        for m in self.res.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, np.sqrt(2))
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.lrelu(self.res(x) + self.shortcut(x))

class DiscBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, dsample):
        super().__init__()
        stride = 2 if dsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
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
            nn.BatchNorm2d(z_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            DiscBlock3(z_dim, z_dim*2,  True),
            nn.Dropout(0.5),
            DiscBlock3(z_dim*2, z_dim*4,  True),
            nn.Dropout(0.5),
            DiscBlock3(z_dim*4, z_dim*8, False),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.disc = nn.Sequential(nn.Linear(z_dim*8, 1), nn.Sigmoid())
        self.ac = nn.Sequential(nn.Linear(z_dim*8, num_class), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        disc_out = self.disc(x).squeeze(1)
        ac_out = self.ac(x)
        return disc_out, ac_out
    

class Generator2(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super(Generator2, self).__init__()
        self.ngf, self.nc, self.nz = z_dim, 128, z_dim
        self.n_classes = num_class

        # condition embedding
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.nc),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz + self.nc, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (rgb channel = 3) x 64 x 64
        )

    def forward(self, noise, labels):
        noise = noise.view(-1, self.nz, 1, 1)
        label_emb = self.label_emb(labels).view(-1, self.nc, 1, 1)
        gen_input = torch.cat((label_emb, noise), 1)
        out = self.main(gen_input)
        return out


class Discriminator2(nn.Module):
    def __init__(self, z_dim=128, num_class=24):
        super(Discriminator2, self).__init__()
        self.ndf = z_dim
        self.n_classes = num_class
        self.main = nn.Sequential(
            # input is (rgb chnannel = 3) x 64 x 64
            nn.Conv2d(3, self.ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*2) x 30 x 30
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*4) x 15 x 15
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*8) x 13 x 13
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (ndf*16) x 7 x 7
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (ndf*32) x 5 x 5
        )
        
        # discriminator fc
        self.fc_dis = nn.Sequential(
            nn.Linear(5*5*self.ndf*32, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc
        self.fc_aux = nn.Sequential(
            nn.Linear(5*5*self.ndf*32, self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv = self.main(input)
        flat = conv.view(-1, 5*5*self.ndf*32)
        fc_dis = self.fc_dis(flat).view(-1, 1).squeeze(1)
        fc_aux = self.fc_aux(flat)
        return fc_dis, fc_aux