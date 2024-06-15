import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import Tensor
import os
import sys
from tqdm import tqdm
from PIL import Image
from dataset import IclevrDataset
from torchvision.utils import save_image, make_grid
from evaluator import evaluation_model
from models.ddpm2 import DDPM, Unet

class Trainer():
    def __init__(self, args, device):
        super(Trainer).__init__()
        train_dataset = IclevrDataset('train', '../iclevr')
        test_dataset = IclevrDataset('test', '../iclevr')
        new_test_dataset = IclevrDataset('new_test', '../iclevr')
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=32, num_workers=args.num_workers)
        self.new_test_loader = DataLoader(new_test_dataset, batch_size=32, num_workers=args.num_workers)

        unet = Unet(in_channels=3, n_feature=args.n_feature, n_classes=24).to(device)
        self.ddpm = DDPM(unet_model=unet, betas=(args.beta_start, args.beta_end), 
                         noise_steps=args.noise_steps, device=device).to(device)

        self.args = args
        self.run_name = args.run_name
        self.lr = args.lr
        self.n_epoch = args.epochs
        self.device = device
        self.evaluator = evaluation_model()

    def train(self):
        optimizer = optim.AdamW(self.ddpm.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.95, patience=5, 
                                                            min_lr=0)
        # lr_scheduler2 = optim.lr_scheduler.LinearLR(optimizer, start_factor= 1,
        #                                            end_factor= 0, total_iters=self.n_epoch)
        best_test_epoch, best_test_new_epoch = 0, 0
        best_test_score, best_test_new_score, best_all = 0, 0, 0
        path_best = os.path.join(self.args.model_root, self.run_name, "model_best.pth")
        path_test = os.path.join(self.args.model_root, self.run_name, "model_test.pth")
        path_test_new = os.path.join(self.args.model_root, self.run_name, "model_test_new.pth")

        for epoch in range(self.n_epoch):
            self.ddpm.train()
            optimizer.param_groups[0]['lr'] = self.lr*(1-epoch/self.n_epoch) # linear lr decay
            
            with tqdm(self.train_loader, leave=True, ncols=100) as pbar:
                pbar.set_description(f"Epoch {epoch}, lr: {lr_scheduler.get_last_lr()[0]:.3e}")
                for x, cond in pbar:
                    optimizer.zero_grad()
                    x = x.to(self.device) # [B,3,64,64]
                    cond = cond.to(self.device) # [B,24]
                    loss = self.ddpm(x, cond)
                    loss.backward()
                    pbar.set_postfix_str(f"loss: {loss:.4f}")
                    optimizer.step()
                pbar.close()
            test_score, test_new_score = self.test_both_save(epoch)

            if not os.path.exists(os.path.join(self.args.model_root, self.run_name)):
                os.makedirs(os.path.join(self.args.model_root, self.run_name))
            if test_score > best_test_score:
                best_test_score = test_score
                best_test_epoch = epoch
                if best_test_score > 0.5:
                    torch.save(self.ddpm.state_dict(), path_test)
            if test_new_score > best_test_new_score:
                best_test_new_score = test_new_score
                best_test_new_epoch = epoch
                if best_test_new_score > 0.5:
                    torch.save(self.ddpm.state_dict(), path_test_new)
            if test_score > 0.8 and test_new_score > 0.8 and test_score + test_new_score > best_all:
                best_all = test_score + test_new_score
                torch.save(self.ddpm.state_dict(), path_best)

            print(f'Current [test:{test_score:.4f}, new test:{test_new_score:.4f}]', end=', ')
            print(f'Best [test:{best_test_score:.4f} @ep{best_test_epoch}, new_test:{best_test_new_score:.4f} @ep{best_test_new_epoch}]')

            # save training model
            path = os.path.join(self.args.model_root, self.run_name, "model_latest_train.pth")
            torch.save(self.ddpm.state_dict(), path)

            # lr scheduler step based on best_test_score + best_test_new_score
            lr_scheduler.step(best_test_score+best_test_new_score)

    def test_both_save(self, epoch):
        test_score, grid1 = self.test(self.test_loader)
        test_new_score, grid2 = self.test(self.new_test_loader)
        if not os.path.exists(os.path.join(self.args.result_root, self.run_name)):
            os.makedirs(os.path.join(self.args.result_root, self.run_name))
        if (epoch+1) % 10 == 0:
            path1 = os.path.join(self.args.result_root, args.run_name, f"test_{epoch}.png")
            path2 = os.path.join(self.args.result_root, args.run_name, f"test_new_{epoch}.png")
        else:
            path1 = os.path.join(self.args.result_root, args.run_name, f"test_current.png")
            path2 = os.path.join(self.args.result_root, args.run_name, f"new_test_current.png")
        save_image(grid1, path1)
        save_image(grid2, path2)
        return test_score, test_new_score
        
    def test(self, test_loader):
        self.ddpm.eval()
        # pbar = tqdm(test_loader, leave=False, ncols=100)
        x_gen, label = [], []
        with torch.no_grad():
            for cond in test_loader:
                cond = cond.to(self.device)
                x_i = self.ddpm.sample(cond, (3, 64, 64), self.device)
                x_gen.append(x_i)
                label.append(cond)
                # score = self.evaluator.eval(x_i, cond)
                # grid = make_grid(x_i, nrow=8, normalize=True)
            x_gen = torch.stack(x_gen, dim=0).squeeze()
            label = torch.stack(label, dim=0).squeeze()
            score = self.evaluator.eval(x_gen, label)
            grid = make_grid(x_gen, nrow=8, normalize=True)
        # pbar.clear()
        return score, grid
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",type=str, choices=["train", "test"], 
                        help="train or test model")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--save_path", default='../ckpt')
    parser.add_argument("--run_name", default='ddpm2')
    parser.add_argument("--inference", default=False, action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    parser.add_argument('--beta_start', default=1e-4, type=float, help='start beta value')
    parser.add_argument('--beta_end', default=0.02, type=float, help='end beta value')
    parser.add_argument('--noise_steps', default=300, type=int, help='frequency of sampling')
    parser.add_argument("--img_size", default=64, type=int, help='image size')
    parser.add_argument('--n_feature', default=256, type=int, 
                        help='time/condition embedding and feature maps dimension')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    
    parser.add_argument("--dataset_path", default="../iclevr", type=str, help="root of dataset dir")
    parser.add_argument("--model_root", default="../ckpt/", type=str, help="model ckpt path")
    parser.add_argument("--result_root", default="../results/", type=str, help="save img path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    if not args.inference:
        if args.resume==True:
            print("Resume training...")
            path = os.path.join(args.model_root, args.run_name, "model_test1_33.pth")
            trainer.ddpm.load_state_dict(torch.load(path))
        print("Start training...")
        trainer.train()  
    else:
        path = os.path.join(args.model_root, args.run_name, "linear/model_best.pth")
        trainer.ddpm.load_state_dict(torch.load(path))
        test_score, grid_test, grid_process = trainer.test(trainer.test_loader)
        path = os.path.join(args.result_root, args.run_name, "test.png")
        path2 = os.path.join(args.result_root, args.run_name, "process.png")
        save_image(grid_test, path)
        save_image(grid_process, path2)

        test_new_score, grid_test_new, _ = trainer.test(trainer.new_test_loader)
        path = os.path.join(args.result_root, args.run_name, "test_new.png")
        save_image(grid_test_new, path)

        print("test acc: {:.4f}, new test acc:{:.4f}".format(test_score, test_new_score))
        print("test done")