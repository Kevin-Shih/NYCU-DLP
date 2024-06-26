import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch import Tensor
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
from tqdm import tqdm
import yaml
import torch.nn.functional as F

class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        MaskGit_CONFIGS["model_param"]["gamma_type"] = args.mask_func
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter
        self.mask_func=args.mask_func
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        args.out = f'../{args.out}_results_{args.sweet_spot}_{args.total_iter}'
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs(args.out, exist_ok=True)
        # os.makedirs(f"{args.out}/final_results_step{args.sweet_spot}", exist_ok=True)
        for step in range(0, args.sweet_spot):
            os.makedirs(f"{args.out}/final_results_step{step}", exist_ok=True)
        os.makedirs(f"{args.out}/mask_scheduling", exist_ok=True)
        os.makedirs(f"{args.out}/imga", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
    def inpainting(self, image, mask_b: Tensor, i): #MakGIT inference
        maska = torch.zeros(self.total_iter, 3, 16, 16) #save all iterations of masks in latent domain
        imga = torch.zeros(self.total_iter+1, 3, 64, 64)#save all iterations of decoded images
        mean = torch.tensor([0.4868, 0.4341, 0.3844],device=self.device).view(3, 1, 1)  
        std = torch.tensor([0.2620, 0.2527, 0.2543],device=self.device).view(3, 1, 1)
        ori=(image[0]*std)+mean
        imga[0]=ori #mask the first image be the ground truth of masked image

        self.model.eval()
        with torch.no_grad():
            _, z_indices = self.model.encode_to_z(image) #z_indices: masked tokens (b,16*16)
            mask_num = mask_b.sum().item() #total number of mask token
            z_indices_predict=z_indices
            mask_bc=mask_b
            mask_b=mask_b.to(device=self.device)
            mask_bc=mask_bc.to(device=self.device)

            ratio = 0
            #iterative decoding for loop design
            #Hint: it's better to save original mask and the updated mask by scheduling separately
            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break
                ratio = step / self.total_iter #this should be updated
                z_indices_predict, mask_bc = self.model.inpainting(ratio, z_indices_predict, mask_bc, mask_num) # predict

                #static method yon can modify or not, make sure your visualization results are correct
                mask_i=mask_bc.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False) #label mask true as black
                mask_image[:, indices[:, 1], indices[:, 2]] = 0 #3,16,16
                maska[step]=mask_image
                shape=(1,16,16,256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img=self.model.vqgan.decode(z_q)
                dec_img_ori=(decoded_img[0]*std)+mean
                imga[step+1]=dec_img_ori #get decoded image

            ##decoded image of the sweet spot only, the test_results folder path will be the --predicted-path for fid score calculation
                vutils.save_image(dec_img_ori, os.path.join(f"{args.out}/final_results_step{step}", f"image_{i:03d}.png"), nrow=1) 
            # vutils.save_image(dec_img_ori, os.path.join(f"{args.out}/final_results", f"image_{i:03d}.png"), nrow=1) 
            #demo score 
            vutils.save_image(maska, os.path.join(f"{args.out}/mask_scheduling", f"test_{i}.png"), nrow=10) 
            vutils.save_image(imga, os.path.join(f"{args.out}/imga", f"test_{i}.png"), nrow=7)



class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device

    def get_mask_latent(self,mask):
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0       #1,3,16*16   check use  
        mask_tokens=(resized_mask[0][0]//1).flatten()   ##[256] =16*16 token
        mask_tokens=mask_tokens.unsqueeze(0)
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0) #true means mask
        return mask_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda:1", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test10_ep67_1.510.pt', help='load ckpt')# 43.30 @ cosine
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test10_ep68_1.736.pt', help='load ckpt')# 44.88 @ cosine
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep45_1.536.pt', help='load ckpt')# 38.95 @ cosine
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep46_1.474.pt', help='load ckpt')# 38.28 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep51_1.415.pt', help='load ckpt')# 37.69 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep52_1.576.pt', help='load ckpt')# 37.94 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep54_1.541.pt', help='load ckpt')# 37.23 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep57_1.491.pt', help='load ckpt')# 36.46 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep64_1.532.pt', help='load ckpt')# 39.61 @ cosine step
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep66_1.448.pt', help='load ckpt')# 36.24 @ cosine step0 39.92 @ cosine step9
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep68_1.402.pt', help='load ckpt')# 36.37 @ cosine step0 39.49 @ cosine step9
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep72_1.451.pt', help='load ckpt')# 35.13 @ cosine step0
    # 38.40 @ cosine step9, 38.30 @ linear step9, 38.43 @ square step9, 37.23 @ log step9
    # 38.59 @ cosine step8, 38.22 @ linear step8, 37.88 @ square step8, 37.31 @ log step8
    # 38.19 @ cosine step7, 38.28 @ linear step7, 37.85 @ square step7, 37.13 @ log step7
    # 37.97 @ cosine step6, 37.72 @ linear step6, 37.63 @ square step6, 37.12 @ log step6
    # 37.42 @ cosine step5, 37.51 @ linear step5, 37.22 @ square step5, 37.18 @ log step5
    # 35.87 @ cosine step4, 36.70 @ linear step4, 36.56 @ square step4, 36.97 @ log step4
    # 35.59 @ cosine step3, 36.40 @ linear step3, 36.12 @ square step3, 36.83 @ log step3
    # 35.48 @ cosine step2, 36.16 @ linear step2, 35.38 @ square step2, 35.13 @ log step2
    # 35.13 @ cosine step1, 35.13 @ linear step1, 35.13 @ square step1, 35.13 @ log step1
    # parser.add_argument('--load-transformer-ckpt-path', type=str, default='../ckpt/tf_test11_ep75_1.519.pt', help='load ckpt')# 37.07 @ cosine step
    
    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='../dataset/cat_face/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='../dataset/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('-t', '--sweet-spot', type=int, default=8, help='sweet spot: the best step in total iteration')  
    parser.add_argument('-T', '--total-iter', type=int, default=8, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='cosine', help='mask scheduling function')
    parser.add_argument('-o', '--out', type=str, default='test', help='output path')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    i=0
    with tqdm(total= len(t.mask_ori), desc=f"Inpainting", ncols=100) as pbar:
        for image, mask in zip(t.mi_ori, t.mask_ori):
            image=image.to(device=args.device)
            mask=mask.to(device=args.device)
            mask_b=t.get_mask_latent(mask)       
            maskgit.inpainting(image,mask_b,i)
            i+=1
            pbar.update(1)
    pbar.close()


