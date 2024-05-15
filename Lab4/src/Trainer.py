import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack
from torch import Tensor
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def calculate_PSNR(gt_image, gen_image):
    PSNR_LIST = []
    for i in range(1, 630):
        PSNR_LIST.append(Generate_PSNR(gt_image[i], gen_image[i][0]).item())
    avg_psnr = sum(PSNR_LIST) / (len(PSNR_LIST) - 1)
    return PSNR_LIST, avg_psnr

def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD

def plot_loss(lists: list, path: str=''):
    length = list(range(len(lists[0])))
    plt.figure(figsize=(8, 6), dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(length, lists[0], label="train", color="blue")
    plt.plot(length, lists[1], label="valid", color="green")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title('Loss Curve (Cyclical)')
    plt.subplot(2, 1, 2)
    plt.plot(length, lists[2], label="teacher forcing")
    plt.plot(length, lists[3], label="beta")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(path)

class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.type = args.kl_anneal_type
        self.ratio = args.kl_anneal_ratio
        self.stop = args.kl_anneal_stop
        self.beta = 0.0
        self.anneal_cycle = args.kl_anneal_cycle
        self.epochs = current_epoch
        if self.type == 'None':
            self.beta = 1.0
    def update(self):
        # TODO
        self.epochs += 1
        if self.type == 'Cyclical':
            self.frange_cycle_linear(stop=self.stop, ratio=self.ratio)
        elif self.type == 'Monotonic':
            self.frange_cycle_linear(stop=self.stop, ratio=1.0)
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, start=0.0, stop=1.0, ratio=1):
        # TODO
        stoped_ep = np.ceil(self.anneal_cycle * ratio)
        slope = (stop - start) / stoped_ep
        mod_epochs = self.epochs if self.type == 'Monotonic' else self.epochs % self.anneal_cycle
        if mod_epochs >= stoped_ep:
            self.beta = stop
        else:
            self.beta = start + mod_epochs * slope
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay= 5e-4)
        self.start_scheduler  = MultiStepLR(self.optim, milestones=[2], gamma=0.1)
        self.scheduler        = ReduceLROnPlateau(self.optim, mode='max', cooldown=5, patience=5, min_lr=1e-8, factor=0.2)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        self.device = args.device
        # self.train_writer = SummaryWriter(f'../log/{args.save_root[8:]}_Train')
        # self.valid_writer = SummaryWriter(f'../log/{args.save_root[8:]}_Valid')

    def forward(self, img, label):
        pass

    def training_stage(self):
        min_loss = 9e-4
        max_psnr = 32
        loss_list = []
        val_loss_list = []
        tf_list = []
        kl_list = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            loss_temp = []
            # adapt_TeacherForcing = True if random.random() < self.tfr else False # mobified after test15 (16 start)
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                adapt_TeacherForcing = True if random.random() < self.tfr else False # mobified after test15 (16 start)
                img = img.to(self.device)
                label = label.to(self.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                loss_temp.append(loss.detach().cpu())

                beta = self.kl_annealing.get_beta()
                TF = 'ON' if adapt_TeacherForcing else 'OFF'
                self.tqdm_bar(f'train [TeacherForcing: {TF}, {self.tfr:.1f}], beta: {beta:.1f}', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            # loss_temp /= len(train_loader)
            # loss_list.append(loss_temp)
            loss_list.append(np.median(np.asarray(loss_temp)))
            kl_list.append(beta)
            tf_list.append(self.tfr)

            self.train(False)
            val_loss, avg_psnr, psnr_list = self.eval()
            self.train(True)
            val_loss_list.append(val_loss)
            # self.train_writer.add_scalar('Loss', loss_temp, self.current_epoch)
            # self.valid_writer.add_scalar('Loss', val_loss, self.current_epoch)
            if (self.current_epoch+1) % self.args.per_save == 0 and (self.current_epoch+1) >= 50:
                plt.close()
                plt.plot(
                np.arange(0, len(psnr_list)),
                    psnr_list,
                    label=f"average PSNR:{avg_psnr:3f}",
                )
                plt.xlabel("Frame index")
                plt.ylabel("PSNR")
                plt.title("Per frame quality (PSNR)")
                plt.legend()
                # plt.show()
                plt.savefig(f"../graphs/Ep{self.current_epoch}_{args.save_root[8:]}_PSNR.jpg")
            # self.valid_writer.add_scalar('PSNR', psnr_list, i)
            if val_loss <= min_loss or avg_psnr >= max_psnr:
                min_loss = val_loss if val_loss < min_loss else min_loss
                max_psnr = avg_psnr if max_psnr < avg_psnr else max_psnr
                if len(self.args.save_name) > 0:
                    self.save(os.path.join(self.args.save_root, f"{self.args.save_name}_best_{val_loss:.4f}_{avg_psnr:.2f}.ckpt"))
                else:
                    self.save(os.path.join(self.args.save_root, f"best_{val_loss:.4f}_{avg_psnr:.2f}.ckpt"))
            # if (self.current_epoch+1) % self.args.per_save == 0 and (self.current_epoch+1) >= 50:
            #     if len(self.args.save_name) > 0:
            #         self.save(os.path.join(self.args.save_root, f"{self.args.save_name}_epoch={self.current_epoch}.ckpt"))
            #     else:
            #         self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            self.current_epoch += 1
            self.scheduler.step(avg_psnr)
            self.start_scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        plot_loss([loss_list, val_loss_list, tf_list, kl_list], path= f'../graphs/{args.save_root[8:]}_LossCurve.jpg')
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.device)
            label = label.to(self.device)
            loss, psnr_list, avg_psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.cpu(), lr=self.scheduler.get_last_lr()[0])
        print(f'Epoch{self.current_epoch} Valid PSNR [max: {np.max(psnr_list):.4f}, avg: {avg_psnr:.4f}, min: {np.min(psnr_list):.4f}]')
        return loss.cpu().item(), avg_psnr, psnr_list
    
    def training_one_step(self,img: Tensor, label: Tensor, adapt_TeacherForcing: bool):
        # TODO
        pred_img = img[:, 0]
        loss = torch.zeros(1, device= self.device)
        beta = self.kl_annealing.get_beta()
        self.optim.zero_grad()
        for i in range(1, self.train_vi_len):
            prev_img = img[:, i - 1] if adapt_TeacherForcing else pred_img

            gt_img_f = self.frame_transformation(prev_img)
            pred_img_f = self.frame_transformation(prev_img)
            label_f = self.label_transformation(label[:, i])

            z, mu, logvar = self.Gaussian_Predictor(gt_img_f, label_f)
            fused_f = self.Decoder_Fusion(pred_img_f, label_f, z)
            pred_img = self.Generator(fused_f)

            loss += self.mse_criterion(pred_img, img[:, i]) + beta * kl_criterion(mu, logvar, self.batch_size)
        loss = loss / (self.train_vi_len - 1)
        if not torch.isnan(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm= 2.0)
            self.optimizer_step()
            return loss
        else:
            loss = None
            return Tensor([np.inf])
        
    
    def val_one_step(self, img, label):
        # TODO
        pred_img = [img[:, 0]]
        loss = torch.zeros(1, device= self.device)
        self.optim.zero_grad()
        for i in range(1, self.val_vi_len):
            pred_img_f = self.frame_transformation(pred_img[i - 1])
            label_f = self.label_transformation(label[:, i])

            z = torch.randn((1, self.args.N_dim, self.args.frame_H, self.args.frame_W), device= self.device)
            fused_f = self.Decoder_Fusion(pred_img_f, label_f, z)
            pred_img.append(self.Generator(fused_f))
            loss += self.mse_criterion(pred_img[i], img[:, i])
        psnr_list, avg_psnr = calculate_PSNR(img[0], pred_img)
        return loss / (self.val_vi_len - 1), psnr_list, avg_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor(),
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader

    def train_dataloader_var(self, ratio=4):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor(),
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len*ratio, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size//ratio,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
  
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.tfr_sde and self.tfr > self.args.tfre:
            self.tfr -= self.tfr_d_step
        if self.tfr < self.args.tfre:
            self.tfr = self.args.tfre
        return
            
    def tqdm_bar(self, mode, pbar: tqdm, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:1.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    if args.device == 'cuda':
        args.device = torch.device('cuda', args.gpu_id)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--lr',             type=float, default=0.001,  help="initial learning rate")
    parser.add_argument('--device',         type=str,   choices=["cuda", "cpu"],    default="cuda")
    parser.add_argument('--optim',          type=str,   choices=["Adam", "AdamW"],  default="Adam")
    parser.add_argument('--gpu',            type=int,   default=1)
    parser.add_argument('--gpu_id', '-g',   type=int,   default=0)
    parser.add_argument('--test',           action='store_true')
    parser.add_argument('--store_visualization',        action='store_true',    help="If you want to see the result while training")
    parser.add_argument('--DR',             type=str,   default='../dataset',   help="Your Dataset Path")
    parser.add_argument('--save_root',      type=str,   default='../ckpt',      help="The path to save your data")
    parser.add_argument('--save_name',      type=str,   default='',        help="The name(prefix) to save your data")
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--num_epoch',      type=int,   default=70,    help="number of total epoch")
    parser.add_argument('--per_save',       type=int,   default=10,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',        type=float, default=1.0,    help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',   type=int,   default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',     type=int,   default=630,    help="valdation video length")
    parser.add_argument('--frame_H',        type=int,   default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',        type=int,   default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim',          type=int,   default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',          type=int,   default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',          type=int,   default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',      type=int,   default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',            type=float,  default=1.0,   help="The initial teacher forcing ratio")
    parser.add_argument('--tfre',           type=float,  default=0.0,   help="The ending teacher forcing ratio")
    parser.add_argument('--tfr_sde',        type=int,    default=5,     help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',     type=float,  default=0.2,   help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',      type=str,    default=None,  help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int,   default=5,      help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str,   choices=["Cyclical", "Monotonic", "None"],  default='Cyclical', help="")
    parser.add_argument('--kl_anneal_cycle',    type=int,   default=10,     help="")
    parser.add_argument('--kl_anneal_stop',     type=float, default=1.0,    help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,    help="")

    args = parser.parse_args()
    main(args)
