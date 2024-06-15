import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam
from torch import Tensor
import os
import sys
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image
from models.acgan import Discriminator2, Generator, Discriminator, Generator2
from dataset import IclevrDataset
from utils import Utils
from evaluator import evaluation_model

class Trainer(object):
    def __init__(self, args):
        self.gen_dim  = args.gdim
        self.disc_dim = args.ddim
        self.num_epochs = args.epochs
        self.diter = args.diter
        self.lambda_ac = args.ac
        self.run_name = args.run_name
        self.save_path = args.save_path

        self.generator = Generator2(z_dim = self.gen_dim).cuda()
        self.discriminator = Discriminator2(z_dim = self.disc_dim).cuda()

        train_dataset = IclevrDataset('train', '../iclevr')
        test_dataset = IclevrDataset('test', '../iclevr')
        new_test_dataset = IclevrDataset('new_test', '../iclevr')
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                      num_workers=args.num_workers)
        self.new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False,
                                          num_workers=args.num_workers)

        self.optimD = AdamW(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimG = AdamW(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.d_sched = ReduceLROnPlateau(self.optimG, mode='max',
                                          factor=0.9, patience=5,
                                          min_lr=0)
        self.g_sched = ReduceLROnPlateau(self.optimG, mode='max',
                                          factor=0.9, patience=5,
                                          min_lr=0)
        self.disc_criterion = nn.BCELoss()
        self.aux_criterion = nn.BCELoss()

    def train_acgan(self):
        gen_iteration = 0
        best_test_epoch, best_test_new_epoch = 0, 0
        best_test_score, best_test_new_score, best_all = 0, 0, 0
        for epoch in range(self.num_epochs):
            iterator = 0
            data_iterator = iter(self.train_loader)
            d_loss, g_loss = 0, 0
            with tqdm(total=len(self.train_loader), ncols=100) as pbar:
                pbar.set_description_str(f'Epoch {epoch}, lr:{self.d_sched.get_last_lr()[0]:.1e}')
                while iterator < len(self.train_loader):
                    if gen_iteration < 25 or gen_iteration % 500 == 0:
                        d_iter_count = 3*self.diter
                    else:
                        d_iter_count = self.diter
                    d_iter = 0
                    # Train the discriminator
                    while d_iter < d_iter_count and iterator < len(self.train_loader):
                        d_iter += 1
                        for p in self.discriminator.parameters():
                            p.requires_grad = True

                        self.discriminator.zero_grad()
                        right_images, aux_label = next(data_iterator)
                        iterator += 1
                        right_images = right_images.float().cuda()
                        aux_label = aux_label.float().cuda()
                        real_label = ((1.0 - 0.7) * torch.rand(aux_label.shape[0]) + 0.7).cuda()
                        fake_label = ((0.3 - 0.0) * torch.rand(aux_label.shape[0]) + 0.0).cuda()
                        if torch.rand(1).item() < 0.1:
                            real_label, fake_label = fake_label, real_label

                        disc_output, pred_real_labels = self.discriminator(right_images)
                        real_loss = self.disc_criterion(disc_output, real_label)

                        noise = torch.randn(right_images.size(0), self.gen_dim).cuda()
                        fake_images = self.generator(noise, aux_label)
                        disc_output, pred_fake_labels = self.discriminator(fake_images)
                        fake_loss = self.disc_criterion(disc_output, fake_label)

                        aux_loss = self.aux_criterion(pred_real_labels, aux_label)
                        aux_loss += self.aux_criterion(pred_fake_labels, aux_label)

                        d_loss = fake_loss + real_loss + self.lambda_ac * aux_loss
                        d_loss.backward()
                        self.optimD.step()

                        pbar.set_postfix_str(f'loss D:{d_loss:.4f}, G:{g_loss:.4f}')
                        pbar.update(1)

                    # Train Generator
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                    self.generator.zero_grad()
                    noise = torch.randn(right_images.size(0), self.gen_dim).cuda()
                    fake_images = self.generator(noise, aux_label)
                    disc_output, pred_gen_labels = self.discriminator(fake_images)
                    aux_loss = self.aux_criterion(pred_gen_labels, aux_label)
                    g_loss = self.disc_criterion(disc_output, fake_label) + self.lambda_ac * aux_loss
                    g_loss.backward()
                    self.optimG.step()

                    pbar.set_postfix_str(f'loss D:{d_loss:.4f}, G:{g_loss:.4f}')
                    pbar.update(0)
                    gen_iteration += 1

            score, score_new = self.predict_both_save(epoch)
            if epoch > 25:
                self.d_sched.step(score+score_new)
                self.g_sched.step(score+score_new)
            if score > best_test_score:
                best_test_score, best_test_epoch = score, epoch
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.run_name, 'test')
            if score_new > best_test_new_score:
                best_test_new_score, best_test_new_epoch = score_new, epoch
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.run_name, 'test_new')
            if score > 0.8 and score_new > 0.8 and score + score_new > best_all:
                best_all = score + score_new
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.run_name, 'best')
            print(f'Current [test:{score:.4f}, new test:{score_new:.4f}]', end=', ')
            print(f'Best [test:{best_test_score:.4f} @ep{best_test_epoch}, new_test:{best_test_new_score:.4f} @ep{best_test_new_epoch}]')

    def predict_both_save(self, epoch):
        if (epoch+1) % 10 == 0:
            path1 = os.path.join(f'../results/{self.run_name}/', f"test_epoch{epoch}.png")
            path2 = os.path.join(f'../results/{self.run_name}/', f"new_test_epoch{epoch}.png")
        else:
            path1 = os.path.join(f'../results/{self.run_name}/', f"test_current.png")
            path2 = os.path.join(f'../results/{self.run_name}/', f"new_test_current.png")
        score, grid = self.predict(self.test_loader)
        save_image(grid, path1)
        score_new, grid_new = self.predict(self.new_test_loader)
        save_image(grid_new, path2)
        return score, score_new

    def predict(self, test_loader):
        self.generator.eval()
        if not os.path.exists(f'../results/{self.run_name}'):
            os.makedirs(f'../results/{self.run_name}')

        with torch.no_grad():
            for text in test_loader:
                noise = torch.randn(32, self.gen_dim).cuda()
                fake_images = self.generator(noise, text.cuda())
                eval_model = evaluation_model()
                score = eval_model.eval(fake_images, text)
                grid = make_grid(fake_images, nrow=8, normalize=True)
        self.generator.train()
        return score, grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--diter", default=1, type=int)
    # parser.add_argument("--gp", default=10, type=int)
    parser.add_argument("--gdim", default=256, type=int)
    parser.add_argument("--ddim", default=64, type=int)
    parser.add_argument("--ac", default=1, type=int)
    parser.add_argument("--cls", default=False, action='store_true')
    parser.add_argument("--save_path", default='../ckpt')
    parser.add_argument("--run_name", default='acgan2')
    parser.add_argument("--inference", default=False, action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    args = parser.parse_args()

    trainer = Trainer(args=args)
    if not args.inference:
        trainer.train_acgan()
    else:
        trainer.generator.load_state_dict(torch.load('../ckpt/acwgan/gen_test.pth'))
        # trainer.discriminator.load_state_dict(torch.load('../ckpt/acwgan/disc_test.pth'))
        score, grid = trainer.predict(trainer.test_loader)
        score_new, grid_new = trainer.predict(trainer.new_test_loader)
        path = os.path.join(f'../results/{args.run_name}/', f"test.png")
        save_image(grid, path)
        path = os.path.join(f'../results/{args.run_name}/', f"new_test.png")
        save_image(grid_new, path)
        print(f'test:{score:.4f}, new_test:{score_new:.4f}')
# test1 use gp=15x, ac=3x