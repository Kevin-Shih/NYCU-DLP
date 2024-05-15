import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from models.resnet34_unet import ResNet34_Unet
# from models.resnet34_unet2 import ResNet34_Unet
from models.unet import UNet 
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import dice_score, dice_loss, WarmUpLR


def train(args, train_dataset, valid_dataset):
    device = torch.device('cuda', args.gpu)
    if 'resnet34' in args.model:
        model = ResNet34_Unet().to(device)
    else:
        model = UNet().to(device)
    if len(args.name) > 0: # create tensorboard writter
        train_writer = SummaryWriter(f'../log/{args.model}_{args.name}_Train')
        valid_writer = SummaryWriter(f'../log/{args.model}_{args.name}_Valid')
    else:
        train_writer = SummaryWriter(f'../log/{args.model}_Train')
        valid_writer = SummaryWriter(f'../log/{args.model}_Valid')

    train_loss = [0 for _ in range(args.epochs)]
    valid_loss = [0 for _ in range(args.epochs)]
    train_dice = [0 for _ in range(args.epochs)]
    valid_dice = [0 for _ in range(args.epochs)]
    best_dice = 0.90
    min_lr = args.lr

    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=  True, num_workers= 8, pin_memory= True)
    valid_loader = DataLoader(valid_dataset, batch_size= args.batch_size, shuffle= False, num_workers= 8, pin_memory= True)
    criterion  = dice_loss
    optimizer = Adam(model.parameters(), lr= args.lr, weight_decay= 5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', cooldown= 3, factor= 0.1, min_lr= 5e-8, patience= 5)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    print( '--------------------------\n'     +
          f'Training {args.model}...\n'       +
          f'Run name: \t{args.name}\n'        +
          f'Total epochs:\t{args.epochs}\n'   +
          f'Batch size:\t{args.batch_size}\n' +
          f'Learning rate:\t{args.lr:1.0E}\n' +
           '--------------------------')
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(total= len(train_loader), desc= f'{epoch + 1:d}/{args.epochs:d}',
                    unit= 'batch', leave= True, ncols= 60, colour= 'cyan')
        for data, mask in train_loader:
            inputs = data.to(device)
            masks = mask.to(device)
            optimizer.zero_grad()
            pred_masks = model.forward(inputs)
            loss = criterion(pred_masks, masks)
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.item()
            train_dice[epoch] += dice_score(pred_masks, masks)
            pbar.update(1)
            if epoch == 0:
                warmup_scheduler.step()
        pbar.close()
        train_loss[epoch] /= len(train_loader)
        train_dice[epoch] /= len(train_loader)
        valid_dice[epoch], valid_loss[epoch] = evaluate(model, valid_loader, device)
        scheduler.step(valid_dice[epoch])

        train_writer.add_scalar('Loss', train_loss[epoch], epoch)
        train_writer.add_scalar('Dice', train_dice[epoch], epoch)
        valid_writer.add_scalar('Loss', valid_loss[epoch], epoch)
        valid_writer.add_scalar('Dice', valid_dice[epoch], epoch)
        print(f'{epoch+1:d}/{args.epochs:d}: ' +
            f'Loss {train_loss[epoch]:6.4f}, Dice {train_dice[epoch]:6.4f} | ' +
            f'Val Loss {valid_loss[epoch]:6.4f}, Dice {valid_dice[epoch]:6.4f}')
        if scheduler.get_last_lr()[-1] < min_lr:
            min_lr = scheduler.get_last_lr()[-1]
            print(f'Adjusting learning rate --> {min_lr:.1E}')

        if valid_dice[epoch] > best_dice:
            best_dice = valid_dice[epoch]
            if len(args.name) > 0:
                torch.save(model.state_dict(), f'../saved_models/{args.model}_{args.name}_{int(best_dice * 100)}.pth')
            else:
                torch.save(model.state_dict(), f'../saved_models/{args.model}_{int(best_dice * 100)}.pth')
    return 

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("-g", "--gpu", help= "gpu id", default= 0, type= int)
    parser.add_argument("-n", "--name", help= "Experiment name", default= "", type= str)
    parser.add_argument('--model', '-m', type=str, default= 'unet', help='target model')
    parser.add_argument('--data_path', '-path', type=str, default= '../dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default= 50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default= 16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default= 1e-4, dest= 'lr', help='learning rate')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train_dataset = load_dataset(args.data_path, 'train')
    valid_dataset = load_dataset(args.data_path, 'valid')
    train(args, train_dataset, valid_dataset)