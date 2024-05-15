import argparse
import torch
from utils import dice_score
from oxford_pet import load_dataset
from models.resnet34_unet import ResNet34_Unet
from models.unet import UNet 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type= int, default= 1, help='batch size')
    parser.add_argument("-g", "--gpu", help= "gpu id", default= 0, type= int)
    return parser.parse_args()

def test(model: nn.Module, test_dataset: TensorDataset, bs: int, device: torch.device) -> float:
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size= bs, num_workers= 8, pin_memory= True)
    test_dice = 0
    with torch.no_grad():
        for data, mask in test_loader:
            inputs = data.to(device)
            masks = mask.to(device).long()
            pred_masks = model.forward(inputs)
            test_dice += dice_score(pred_masks, masks)
        test_dice /= len(test_loader)
    return test_dice

if __name__ == '__main__':
    args = get_args()
    test_dataset = load_dataset('../dataset', 'test')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if 'resnet34' in args.model:
        model = ResNet34_Unet().to(device)
        model.load_state_dict(torch.load(args.model))
        test_dice = test(model, test_dataset, args.batch_size, device)
        print(f'ResNet34_UNet:\tTest Dice {test_dice:.4f}')
    else:
        model = UNet().to(device)
        model.load_state_dict(torch.load(args.model))
        test_dice = test(model, test_dataset, args.batch_size, device)
        print(f'UNet:\tTest Dice {test_dice:.4f}')
