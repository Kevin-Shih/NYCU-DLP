import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
from torchvision import transforms

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_residual=False):
        super().__init__()
        self.same_channels = (in_channels==out_channels) 
        self.is_residual = is_residual

        # conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_residual:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

"""down sampling image feature maps"""
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        out = self.model(x)
        return out

"""up sampling image feature maps"""
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        out = torch.cat((x, skip), 1)
        out = self.model(out)
        return out

"""embed time and label condition"""
class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed, self).__init__()

        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        out = self.model(out)
        return out

"""Unet model"""
class Unet(nn.Module):
    def __init__(self, in_channels, n_feature=256, n_classes=24):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feature = n_feature
        self.n_classes = n_classes

        """conv first"""
        self.initial_conv = ResidualConvBlock(in_channels, n_feature, is_residual=True)
         
        """down sampling"""
        self.down1 = DownSample(n_feature, n_feature)
        self.down2 = DownSample(n_feature, 2 * n_feature)

        """bottom hidden of unet"""
        self.hidden = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        """embed time and condition"""
        self.time_embed1 = Embed(1, 2*n_feature)
        self.time_embed2 = Embed(1, 1*n_feature)
        self.cond_embed1 = Embed(n_classes, 2*n_feature)
        self.cond_embed2 = Embed(n_classes, 1*n_feature)

        self.time_embed_down1 = Embed(1, 1*n_feature)
        self.cond_embed_down1 = Embed(n_classes, 1*n_feature)
        self.time_embed_down2 = Embed(1, 1*n_feature)
        self.cond_embed_down2 = Embed(n_classes, 1*n_feature)

        """up sampling (choose to concat embedding at hidden or not)"""
        self.up0 = nn.Sequential(
            # if: concat time embedding and condition embedding (6*n_feature)
            # nn.ConvTranspose2d(6 * n_feature, 2 * n_feature, 8, 8), 
            # else: multiply & add embedding later (2*n_feature)
            nn.ConvTranspose2d(2 * n_feature, 2 * n_feature, 8, 8), 
            nn.GroupNorm(8, 2 * n_feature),
            nn.ReLU(True),
        )
        self.up1 = UpSample(4 * n_feature, n_feature)
        self.up2 = UpSample(2 * n_feature, n_feature)

        """output"""
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feature, n_feature, 3, 1, 1),
            nn.GroupNorm(8, n_feature),
            nn.ReLU(True),
            nn.Conv2d(n_feature, self.in_channels, 3, 1, 1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, cond, time):
        # embed context, time step
        cond_emb1 = self.cond_embed1(cond).view(-1, self.n_feature * 2, 1, 1) # [32,512,1,1]
        time_emb1 = self.time_embed1(time).view(-1, self.n_feature * 2, 1, 1) # [32,512,1,1]
        cond_emb2 = self.cond_embed2(cond).view(-1, self.n_feature, 1, 1) # [32,256,1,1]
        time_emb2 = self.time_embed2(time).view(-1, self.n_feature, 1, 1) # [32,256,1,1]
        # for down
        cond_emb_down1 = self.cond_embed_down1(cond).view(-1, self.n_feature, 1, 1) 
        time_emb_down1 = self.time_embed_down1(time).view(-1, self.n_feature, 1, 1)
        cond_emb_down2 = self.cond_embed_down2(cond).view(-1, self.n_feature, 1, 1) 
        time_emb_down2 = self.time_embed_down2(time).view(-1, self.n_feature, 1, 1)

        # initial conv
        x = self.initial_conv(x)  # [32,256,64,64]

        # down sampling
        down1 = self.down1(cond_emb_down1*x+ time_emb_down1)
        down2 = self.down2(cond_emb_down2*down1+ time_emb_down2)

        # hidden
        hidden = self.hidden(down2)

        # choose to concatenate the embedding at hidden or not
        # hidden = torch.cat((hidden, temb1, cemb1), 1)

        # up sampling
        up1 = self.up0(hidden) # [32,256,64,64]
        up2 = self.up1(cond_emb1*up1+ time_emb1, down2)  
        up3 = self.up2(cond_emb2*up2+ time_emb2, down1)

        # output
        out = self.out(torch.cat((up3, x), 1))
        return out

from utils import ddpm_schedules


class DDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device):
        super(DDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], noise_steps).items():
            self.register_buffer(k, v)

        # loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x, cond):
        """training ddpm, sample time and noise randomly (return loss)"""
        # t ~ Uniform(0, n_T)
        timestep = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        # eps ~ N(0, 1)
        noise = torch.randn_like(x)  

        x_t = (
            self.sqrtab[timestep, None, None, None] * x
            + self.sqrtmab[timestep, None, None, None] * noise
        ) 

        predict_noise = self.unet_model(x_t, cond, timestep/self.n_T)

        # return MSE loss between real added noise and predicted noise
        # loss = self.mse_loss(noise, predict_noise)
        loss = self.mse_loss(noise, predict_noise)
        return loss

    def sample(self, cond, size, device):
        """sample initial noise and generate images based on conditions"""
        n_sample = len(cond)
        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)
        x_seq = x_i[0:1]
        save_idx = [250, 200, 150, 100, 75, 50, 30, 15, 0]
        with tqdm(range(self.n_T, 0, -1), leave=False, desc='Sampling', ncols=100) as pbar:
            for idx in pbar:
                timestep = torch.tensor([idx / self.n_T]).to(device)
                z = torch.randn(n_sample, *size).to(device) if idx > 1 else 0
                eps = self.unet_model(x_i, cond, timestep)
                x_i:torch.Tensor = (
                    self.oneover_sqrta[idx] * (x_i - eps * self.mab_over_sqrtmab[idx])
                    + self.sqrt_beta_t[idx] * z
                )
                if (idx-1) in save_idx:
                    x_t = x_i[0:1]
                    x_seq = torch.cat((x_seq, x_t), dim=0)
        pbar.close()
        return x_i, x_seq