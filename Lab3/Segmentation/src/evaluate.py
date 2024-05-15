import torch
import torch.nn as nn
from utils import dice_score, dice_loss

def evaluate(model: nn.Module, valid_loader, device):
    model.eval()
    valid_dice = 0
    valid_loss = 0
    criterion  = dice_loss
    with torch.no_grad():
        for data, mask in valid_loader:
            inputs = data.to(device)
            masks = mask.to(device)
            pred_masks = model.forward(inputs)
            valid_loss += criterion(pred_masks, masks)
            valid_dice += dice_score(pred_masks, masks)
        valid_dice /= len(valid_loader)
        valid_loss /= len(valid_loader)
    return valid_dice, valid_loss