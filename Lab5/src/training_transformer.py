import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData, WarmUpLR
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        if args.data_parallel:
            args.batch_size *= 2
            self.model = nn.DataParallel(VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device))
        else:
            self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.load_data(args)
        self.optim, self.scheduler = self.configure_optimizers()
        self.epoch = 0
        os.makedirs(args.ckpt_root, exist_ok=True)
        if len(args.start_from_ckpt) > 0:
            path = os.path.join(args.ckpt_root, args.start_from_ckpt)
            self.model.load_transformer_checkpoint(path)
            print(f"Loaded Transformer from {path}.")

    def train(self):
        self.step = 0
        for self.epoch in range(1, args.epochs+1):
            self._train_one_epoch()
            valid_loss = self._eval_one_epoch()
            # print(f'Avg. Validation loss = {valid_loss:.4f}')
            
            if self.epoch % args.ckpt_interval == 0:
                tf_statedict = self.model.module.transformer.state_dict() if args.data_parallel else self.model.transformer.state_dict()
                name = f'{args.name}_epoch{self.epoch}' if len(args.name) > 0 else f'epoch{self.epoch}'
                torch.save(tf_statedict, os.path.join(args.ckpt_root, f"tf_{name}.pt"))
                torch.save(self.model.state_dict(), os.path.join(args.ckpt_root, f"maskgit_{name}.pt"))
            # torch.save(self.model.module.transformer.state_dict(), os.path.join(args.ckpt_root, "transformer_current.pt"))
            # torch.save(self.model.transformer.state_dict(), os.path.join(args.ckpt_root, "transformer_current.pt"))
            # torch.save(self.model.state_dict(), os.path.join(args.ckpt_root, "maskgit_current.pt"))


    def _train_one_epoch(self):
        with tqdm(total= len(self.train_loader), desc=f"Train Epoch {self.epoch}", ncols=100) as pbar:
            for imgs in self.train_loader:
                if not args.data_parallel:
                    imgs = imgs.to(device=args.device)
                logits, z_indices = self.model(imgs)
                # print()
                # print(logits.shape, z_indices.shape)
                # print(logits.reshape(-1, logits.size(-1)).shape, z_indices.reshape(-1).shape)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                loss.backward()
                if self.step % args.accum_grad == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                if self.epoch <= args.warmup_epoch:
                    self.scheduler.step()
                self.step += 1
                pbar.set_postfix_str(f'Loss={loss.cpu().detach().item():.4f}')
                pbar.update(1)
        pbar.close()

    @torch.no_grad
    def _eval_one_epoch(self):
        self.model.eval()
        loss_all=0
        with tqdm(total= len(self.val_loader), desc=f"Valid Epoch {self.epoch}", ncols=100) as pbar:
            for imgs in self.val_loader:
                if not args.data_parallel:
                    imgs = imgs.to(device=args.device)
                logits, z_indices = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                loss_all+=loss.item()
                pbar.update(1)
        avg_loss = loss_all/len(self.val_loader)
        pbar.set_postfix_str(f'Avg. Loss={avg_loss:.4f}')
        pbar.refresh()
        pbar.close()
        self.model.train()
        return avg_loss

    def configure_optimizers(self):
        if args.data_parallel:
            optimizer = torch.optim.Adam(self.model.module.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        else:
            optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = WarmUpLR(
            optimizer=optimizer,
            warmup_epoch= args.warmup_epoch,
            total_iters= len(self.train_loader)
        )
        return optimizer,scheduler

    def load_data(self, args):
        train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
        self.train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    drop_last=True,
                                    pin_memory=True,
                                    shuffle=True)
        
        val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
        self.val_loader =  DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    drop_last=True,
                                    pin_memory=True,
                                    shuffle=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct -> Checked
    parser.add_argument('--train_d_path', type=str, default="../dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="../dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--ckpt_root', type=str, default='../ckpt', help='Path to checkpoint dir.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--data-parallel', action='store_true', help='Use DataParallel.')
    parser.add_argument('--partial', type=float, default=1.0, help='Data used in training')
    parser.add_argument('--name', type=str, default='', help='Run name')
    parser.add_argument('--accum-grad', type=int, default=6, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_ckpt', type=str, default='', help='start_from_ckpt.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='warmup_epoch.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    trainer = TrainTransformer(args, MaskGit_CONFIGS)
    trainer.train()