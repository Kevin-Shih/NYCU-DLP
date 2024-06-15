import numpy as np
from torch import Tensor, nn
from torch import  autograd
import torch
import os

class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod
    def gradient_penalty(D, real, fake):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(real.device)
        alpha = alpha.expand(real.size())
        interpolates:Tensor = alpha * real + (1-alpha) * fake
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        interpolates_d, _ = D(interpolates)
        gradients = autograd.grad(
            inputs=interpolates,
            outputs=interpolates_d,
            grad_outputs=torch.ones_like(interpolates_d),
            create_graph=True,
            retain_graph=True,
        )[0]
        normalised_grad = torch.norm(gradients.view(batch_size, -1), 2, dim=1)
        gp = torch.mean((normalised_grad - 1) ** 2)
        return gp

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

def ddpm_schedules(beta1, beta2, T, schedule_type="linear"):

    # assert (beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)")

    # if schedule_type == "linear":
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_B = torch.sqrt(beta_t)

    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    a_bar = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_Abar = torch.sqrt(a_bar)
    one_over_sqrt_A = 1 / torch.sqrt(alpha_t)

    sqrt_mAbar = torch.sqrt(1 - a_bar)
    mA_over_sqrt_mAbar = (1 - alpha_t) / sqrt_mAbar

    return {
        "alpha_t": alpha_t,  # alpha_t
        # "beta": beta_t,  # alpha_t
        "oneover_sqrta": one_over_sqrt_A,  # 1/sqrt{alpha_t}
        "sqrt_beta_t": sqrt_B,  # sqrt{beta_t}
        "alphabar_t": a_bar,  # bar{alpha_t}
        "sqrtab": sqrt_Abar,  # sqrt{bar{alpha_t}}
        "sqrtmab": sqrt_mAbar,  # sqrt{1-bar{alpha_t}}
        "mab_over_sqrtmab": mA_over_sqrt_mAbar,  # (1-alpha_t)/sqrt{1-bar{alpha_t}}
    }

# for ddpm2
def ddpm_schedules2(beta1, beta2, T, schedule_type="linear"):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_B = torch.sqrt(beta_t)

    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    a_bar = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_Abar = torch.sqrt(a_bar)
    one_over_sqrt_A = 1 / torch.sqrt(alpha_t)

    sqrt_mAbar = torch.sqrt(1 - a_bar)
    mA_over_sqrt_mAbar = (1 - alpha_t) / sqrt_mAbar

    return {
        "alpha": alpha_t,  # alpha_t
        "beta": beta_t,  # alpha_t
        "one_over_sqrt_A": one_over_sqrt_A,  # 1/sqrt{alpha_t}
        "sqrt_B": sqrt_B,  # sqrt{beta_t}
        "a_bar": a_bar,  # bar{alpha_t}
        "sqrt_Abar": sqrt_Abar,  # sqrt{bar{alpha_t}}
        "sqrt_mAbar": sqrt_mAbar,  # sqrt{1-bar{alpha_t}}
        "mA_over_sqrt_mAbar": mA_over_sqrt_mAbar,  # (1-alpha_t)/sqrt{1-bar{alpha_t}}
    }