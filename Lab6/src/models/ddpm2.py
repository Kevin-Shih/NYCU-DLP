import torch
import torch.nn as nn
from tqdm import tqdm
from utils import ddpm_schedules2 as ddpm_schedules

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.no_short = (in_channels==out_channels and not downsample)
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GELU(),
        )
        self.short = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GELU(),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        # adds on correct residual in case channels have increased
        if self.no_short:
            out = x + x1
        else:
            out = self.short(x) + x1
        return self.gelu(out) / 1.414

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        # conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 2 if downsample else 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        return x1

"""down sampling image feature maps"""
""" class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.model = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride= True),
            # nn.MaxPool2d(2)
        )

    def forward(self, x):
        out = self.model(x)
        return out """

"""up sampling image feature maps"""
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ConvBlock(out_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        out = torch.cat((x, skip), 1)
        out = self.model(out)
        return out

"""embed time and label condition"""
class Embed(nn.Module):
    def __init__(self, input_dim=24, input_dim2=1, emb_dim=128):
        super(Embed, self).__init__()

        self.input_dim = input_dim
        self.input_dim2 = input_dim2
        self.emb_dim = emb_dim
        self.model1 = nn.Sequential( # class
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.model2 = nn.Sequential( # time
            nn.Linear(input_dim2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, y):
        x = x.view(-1, self.input_dim)
        y = y.view(-1, self.input_dim2)
        out1 = self.model1(x).view(-1, self.emb_dim, 1, 1)
        out2 = self.model2(y).view(-1, self.emb_dim, 1, 1)
        return out1, out2

"""Unet model"""
class Unet(nn.Module):
    def __init__(self, in_channels, n_feature=256, n_classes=24):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_feature = n_feature
        self.n_classes = n_classes

        self.initial_conv = ResidualConvBlock(in_channels, n_feature)
         
        self.down1 = ConvBlock(n_feature, n_feature, downsample= True)
        self.down2 = ConvBlock(n_feature, 2 * n_feature, downsample= True)

        self.bridge = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.GELU())
        # self.bridge = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.GELU())

        self.embed_u1 = Embed(n_classes, 1, 2*n_feature)
        self.embed_u2 = Embed(n_classes, 1, 1*n_feature)
        self.embed_d1 = Embed(n_classes, 1, 1*n_feature)
        self.embed_d2 = Embed(n_classes, 1, 1*n_feature)

        self.up0 = nn.Sequential(
            # if: concat time embedding and condition embedding (6*n_feature)
            # nn.ConvTranspose2d(6 * n_feature, 2 * n_feature, 8, 8), 
            # else: multiply & add embedding later (2*n_feature)
            nn.ConvTranspose2d(2 * n_feature, 2 * n_feature, 8, 8), # (2-1) * 8 + 1*(8-1) +1 = 16
            # nn.ConvTranspose2d(2 * n_feature, 2 * n_feature, 4, 4), # (4-1) * 4 + 1*(4-1) +1 = 16
            nn.GroupNorm(8, 2 * n_feature),
            nn.ReLU(True),
        )
        self.up1 = UpSample(4 * n_feature, n_feature)
        self.up2 = UpSample(2 * n_feature, n_feature)

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
        cond_emb_u1, time_emb_u1 = self.embed_u1(cond, time) # [32,512,1,1]
        cond_emb_u2, time_emb_u2 = self.embed_u2(cond, time) # [32,256,1,1]
        # for down
        cond_emb_d1, time_emb_d1 = self.embed_d1(cond, time)
        cond_emb_d2, time_emb_d2 = self.embed_d2(cond, time)

        # down sampling
        x = self.initial_conv(x)  # [32,256,64,64]
        down1 = self.down1(cond_emb_d1*x + time_emb_d1) # [32,256,32,32]
        down2 = self.down2(cond_emb_d2*down1 + time_emb_d2) # [32,512,16,16]
        bridge = self.bridge(down2) # [32,512,2,2]
        # up sampling
        up1 = self.up0(bridge) # [32,256,16,16]
        up2 = self.up1(cond_emb_u1*up1+ time_emb_u1, down2)  # [32,256,32,32]
        up3 = self.up2(cond_emb_u2*up2+ time_emb_u2, down1)  # [32,256,64,64]
        
        # output
        out = self.out(torch.cat((up3, x), 1))
        # print(bridge.shape, up1.shape, up3.shape, out.shape)
        return out

class DDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device):
        super(DDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model
        self.mse_loss = nn.MSELoss()
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], noise_steps).items():
            self.register_buffer(k, v)       

    def forward(self, x, cond):
        """training ddpm, sample time and noise randomly (return loss)"""
        timestep = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x) # eps ~ N(0, 1)
        x_t = self.sqrt_Abar[timestep, None, None, None] * x + self.sqrt_mAbar[timestep, None, None, None] * noise
        predict_noise = self.unet_model(x_t, cond, timestep/self.n_T)

        loss = self.mse_loss(noise, predict_noise) # unet mse loss
        return loss

    def sample(self, cond, size, device):
        """sample initial noise and generate images based on conditions"""
        n_sample = len(cond)
        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)  
        with tqdm(range(self.n_T, 0, -1), leave=False, desc='Sampling', ncols=100) as pbar:
            for idx in pbar:
                timestep = torch.tensor([idx / self.n_T]).to(device)
                z = torch.randn(n_sample, *size).to(device) if idx > 1 else 0
                eps = self.unet_model(x_i, cond, timestep)
                x_i = (self.one_over_sqrt_A[idx] * (x_i - eps * self.mA_over_sqrt_mAbar[idx])
                       + self.sqrt_B[idx] * z)
        pbar.close()
        return x_i