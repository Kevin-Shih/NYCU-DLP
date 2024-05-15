from torch import Tensor, no_grad
from torch.optim.lr_scheduler import _LRScheduler

def dice_score(pred_mask :Tensor, gt_mask: Tensor):
    with no_grad():
        pred_mask[pred_mask >= 0.5] = 1
        pred_mask[pred_mask < 0.5] = 0
        smooth = 1e-6
        inter = (pred_mask * gt_mask).sum()
        return (2.0 * inter + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

def dice_loss(pred_mask :Tensor, gt_mask: Tensor):
    smooth = 1e-6
    inter = (pred_mask * gt_mask).sum()
    return 1 - (2.0 * inter + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters= 197, last_epoch= -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (1e-8 if self.total_iters == 0 else self.total_iters) for base_lr in self.base_lrs]
    
class EarlyStopping:
    def __init__(self, patience= 10, min_delta= 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_valid_loss = float('inf')

    def __call__(self, valid_loss):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss > (self.min_valid_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False