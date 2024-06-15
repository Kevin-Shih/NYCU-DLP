import os
import argparse
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import AdamW, Adam
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from models.acwgan_gp import Discriminator4, Generator4
from dataset import IclevrDataset
from utils import Utils
from evaluator import evaluation_model

class Trainer(object):
    def __init__(self, args):
        self.gen_dim  = args.gdim
        self.disc_dim = args.ddim
        # acwgangp2 = gen3, disc3
        # acwgangp3 = gen3, disc4
        # acwgangp4 = gen4, disc4
        # acwgangp3a = gen3a, disc4
        # acwgangp3aa = gen3a, disc4a
        # acwgangp4a = gen4a, disc4
        # acwgangp4aa = gen4a, disc4a
        # acwgangp4ab = gen4a, disc4b
        # acwgangp4b = gen4, disc4b
        # self.generator = Generator3(z_dim= self.gen_dim).cuda()
        self.generator = Generator4(z_dim= self.gen_dim).cuda()
        # self.generator = Generator4a(z_dim= self.gen_dim).cuda()
        # self.discriminator = Discriminator3(z_dim= self.disc_dim).cuda()
        self.discriminator = Discriminator4(z_dim= self.disc_dim).cuda()
        # self.discriminator = Discriminator4a(z_dim= self.disc_dim).cuda()
        # self.discriminator = Discriminator4b(z_dim= self.disc_dim).cuda()
        self.eval_model = evaluation_model()

        self.train_dataset = IclevrDataset('train', '../iclevr')
        test_dataset = IclevrDataset('test', '../iclevr')
        new_test_dataset = IclevrDataset('new_test', '../iclevr')
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.beta1 = 0.5 #0.5
        self.num_epochs = args.epochs
        self.diter = args.diter
        self.lambda_gp = args.gp
        self.lambda_ac_g = args.ac_g
        self.lambda_ac_d = args.ac_d

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                      num_workers=self.num_workers)
        self.new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False,
                                          num_workers=self.num_workers)

        self.optimD = AdamW(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = AdamW(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # self.d_sched = LambdaLR(self.optimD, lambda epcoh: 1 - epcoh / self.num_epochs)
        # self.g_sched = LambdaLR(self.optimG, lambda epcoh: 1 - epcoh / self.num_epochs)
        self.d_sched2 = ReduceLROnPlateau(self.optimG, mode='max',
                                          factor=0.9, cooldown=10,
                                          patience=5, min_lr=0)
        self.g_sched2 = ReduceLROnPlateau(self.optimG, mode='max',
                                          factor=0.9, cooldown=10,
                                          patience=5, min_lr=0)
        self.run_name = args.run_name
        self.save_path = args.save_path

    def train_wgan(self):
        gen_iteration = 0
        best_test_epoch, best_test_new_epoch = 0, 0
        best_test_score, best_test_new_score, best_all = 0, 0, 0
        for epoch in range(self.num_epochs):
            iterator = 0
            data_iterator = iter(self.train_loader)
            d_loss, g_loss = 0, 0
            with tqdm(total=len(self.train_loader), ncols=100) as pbar:
                pbar.set_description_str(f'Epoch {epoch}, lr:{self.d_sched2.get_last_lr()[0]:.1e}')
                while iterator < len(self.train_loader):
                    d_iter = 0
                    # Train the discriminator
                    while d_iter < self.diter and iterator < len(self.train_loader):
                        d_iter += 1
                        for p in self.discriminator.parameters():
                            p.requires_grad = True

                        self.discriminator.zero_grad()
                        right_images, right_label = next(data_iterator)
                        iterator += 1

                        right_images = Variable(right_images.float().cuda())
                        right_label = Variable(right_label.float().cuda())
                        d_loss_real, pred_real_labels = self.discriminator(right_images)
                        real_loss = torch.mean(d_loss_real)

                        noise = torch.randn(right_images.size(0), self.gen_dim).cuda()
                        fake_images = self.generator(noise, right_label)
                        d_loss_fake, pred_fake_labels = self.discriminator(fake_images)
                        fake_loss = torch.mean(d_loss_fake)

                        gp = Utils.gradient_penalty(self.discriminator, right_images, fake_images)
                        aux_loss = torch.nn.functional.binary_cross_entropy(
                            torch.cat((pred_real_labels, pred_fake_labels), 0),
                            torch.cat((right_label, right_label), 0),
                        )
                        d_loss:Tensor = (fake_loss - real_loss) + self.lambda_ac_d * aux_loss + self.lambda_gp * gp
                        d_loss.backward()
                        self.optimD.step()

                        pbar.set_postfix_str(f'loss D:{d_loss:.4f}, G:{g_loss:.4f}')
                        pbar.update(1)

                    # Train Generator
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                    self.generator.zero_grad()
                    noise = torch.randn(right_images.size(0), self.gen_dim).cuda()
                    fake_images = self.generator(noise, right_label)
                    g_loss, pred_gen_labels = self.discriminator(fake_images)

                    g_loss = -g_loss.mean()
                    aux_loss = nn.functional.binary_cross_entropy(pred_gen_labels, right_label)
                    g_loss += self.lambda_ac_g * aux_loss
                    g_loss.backward()
                    self.optimG.step()
                    pbar.set_postfix_str(f'loss D:{d_loss:.4f}, G:{g_loss:.4f}')
                    pbar.update(0)
                    gen_iteration += 1
            score, score_new = self.predict_both_save(epoch)
            # self.d_sched.step()
            # self.g_sched.step()
            # if epoch > 10:
            #     self.d_sched2.step(score+score_new)
            #     self.g_sched2.step(score+score_new)
            if score >= best_test_score:
                best_test_score, best_test_epoch = score, epoch
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.run_name, 'test')
            if score_new >= best_test_new_score:
                best_test_new_score, best_test_new_epoch = score_new, epoch
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.run_name, 'test_new')
            if score >= 0.5 and score_new >= 0.5 and score + score_new >= best_all:
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
            for cond in test_loader:
                noise = torch.randn(32, self.gen_dim).cuda()
                fake_images = self.generator(noise, cond.cuda())
                # eval_model = evaluation_model()

                score = self.eval_model.eval(fake_images, cond)
                grid = make_grid(fake_images, nrow=8, normalize=True)
        self.generator.train()
        return score, grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--diter", default=1, type=int)
    parser.add_argument("--gdim", default=256, type=int)
    parser.add_argument("--ddim", default=64, type=int)
    parser.add_argument("--gp", default=10, type=int)
    parser.add_argument("--ac_d", default=1, type=int)
    parser.add_argument("--ac_g", default=1, type=int)
    # parser.add_argument("--cls", default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    parser.add_argument("--save_path", default='../ckpt')
    parser.add_argument("--run_name", default='test')
    parser.add_argument("--inference", default=False, action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    args = parser.parse_args()

    trainer = Trainer(args=args)
    if not args.inference:
        if args.resume==True:
            print("Resume training...")
            # gen_path = os.path.join(args.save_path, "acwgangp4/gen_test_new_ep333.pth")
            # disc_path = os.path.join(args.save_path, "acwgangp4/disc_test_new_ep333.pth")
            # gen_path = os.path.join(args.save_path, "acwgangp4a/gen_test_new_ep89.pth")
            # disc_path = os.path.join(args.save_path, "acwgangp4a/disc_test_new_ep89.pth")
            gen_path = os.path.join(args.save_path, "acwgangp4b/gen_test_new_ep158.pth")
            disc_path = os.path.join(args.save_path, "acwgangp4b/disc_test_new_ep158.pth")
            trainer.generator.load_state_dict(torch.load(gen_path))
            trainer.discriminator.load_state_dict(torch.load(disc_path))
        trainer.train_wgan()
    else:
        # trainer.generator.load_state_dict(torch.load('../ckpt/acwgangp4_ac0.5/gen_best_ep26.pth')) #
        trainer.generator.load_state_dict(torch.load('../ckpt/acwgangp4_ac0.5/gen_best.pth')) # best for acwgangp4 use this
        # trainer.generator.load_state_dict(torch.load('../ckpt/acwgangp4b_equal/gen_best_ep36.pth')) # best
        # trainer.generator.load_state_dict(torch.load('../ckpt/acwgangp4b_equal/gen_test_new.pth'))
        # trainer.discriminator.load_state_dict(torch.load('../ckpt/acwgangp/disc_test.pth'))
        score, grid = trainer.predict(trainer.test_loader)
        score_new, grid_new = trainer.predict(trainer.new_test_loader)
        path = os.path.join(f'../results/{args.run_name}/', f"test.png")
        save_image(grid, path)
        path = os.path.join(f'../results/{args.run_name}/', f"new_test.png")
        save_image(grid_new, path)
        print(f'test:{score:.4f}, new_test:{score_new:.4f}')
# test1 use gp=15x, ac=3x