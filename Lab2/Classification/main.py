import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
from dataloader import ButterflyMothLoader
from ResNet50 import ResNet50
from VGG19 import VGG19

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

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters= 197, last_epoch= -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (1e-8 if self.total_iters == 0 else self.total_iters) for base_lr in self.base_lrs]

def evaluate(model: nn.Module, valid_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    valid_acc = 0
    valid_loss = 0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, label in valid_loader:
            inputs = data.to(device)
            labels = label.to(device)
            pred_labels = model.forward(inputs)
            valid_loss += criterion(pred_labels, labels)
            valid_acc += (torch.max(pred_labels, 1)[1] == labels).sum().item()
        valid_acc *= 100.0 / len(valid_loader.dataset)
        valid_loss /= len(valid_loader)
    return valid_acc, valid_loss

def test(model: nn.Module, test_dataset: TensorDataset, bs: int) -> float:
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size= bs, num_workers= 8, pin_memory= True)
    test_acc = 0
    with torch.no_grad():
        for data, label in test_loader:
            inputs = data.to(device)
            labels = label.to(device).long()
            pred_labels = model.forward(inputs)
            test_acc += (torch.max(pred_labels, 1)[1] == labels).sum().item()
        test_acc *= 100.0 / len(test_dataset)
    return test_acc

def train(target_model: str, train_dataset: TensorDataset, valid_dataset: TensorDataset,
          epochs: int, lr: float, bs: int, device: torch.device, run_name: str= ''):
    if len(run_name) > 0:
        train_writer = SummaryWriter(f'log/{target_model}_{run_name}_Train')
        valid_writer = SummaryWriter(f'log/{target_model}_{run_name}_Valid')
    else:
        train_writer = SummaryWriter(f'log/{target_model}_Train')
        valid_writer = SummaryWriter(f'log/{target_model}_Valid')
    if 'VGG' in target_model:
        model = VGG19().to(device)
    else:
        model = ResNet50().to(device)

    train_loss = [0 for _ in range(epochs)]
    valid_loss = [0 for _ in range(epochs)]
    train_acc = [0 for _ in range(epochs)]
    valid_acc = [0 for _ in range(epochs)]
    best_acc = 85
    min_lr = lr

    train_loader = DataLoader(train_dataset, batch_size= bs, shuffle= True, num_workers= 8, pin_memory= True)
    valid_loader = DataLoader(valid_dataset, batch_size= bs, shuffle= False, num_workers= 8, pin_memory= True)
    criterion  = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= lr, weight_decay= 5e-4)
    scheduler = MultiStepLR(optimizer, milestones= [60, 120, 160], gamma= 0.2)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    print( '--------------------------\n'+
          f'Training {target_model}...\n'+
          f'Run name: \t{run_name}\n'      +
          f'Total epochs:\t{epochs}\n'   +
          f'Batch size:\t{batch_size}\n' +
          f'Learning rate:\t{lr}\n'      +
           '--------------------------')
    for epoch in range(epochs):
        # Train model
        model.train()
        pbar = tqdm(total= len(train_loader), desc= f'[{epoch + 1:3d}/{epochs:3d}]',
                    unit= 'batch', leave= True, ncols= 65, colour= 'cyan')
        for data, label in train_loader:
            inputs = data.to(device)
            labels = label.to(device)
            optimizer.zero_grad()
            pred_labels = model.forward(inputs)
            loss = criterion(pred_labels, labels)
            loss.backward()
            optimizer.step()

            train_loss[epoch] += loss.item()
            train_acc[epoch] += (torch.max(pred_labels, 1)[1] == labels).sum().item()
            if epoch == 0:
                warmup_scheduler.step()
            pbar.update(1)
        pbar.close()

        valid_acc[epoch], valid_loss[epoch] = evaluate(model, valid_loader, device)
        train_loss[epoch] /= len(train_loader)
        train_acc[epoch] *= 100.0 / len(train_dataset)
        scheduler.step()

        train_writer.add_scalar('Loss', train_loss[epoch], epoch)
        train_writer.add_scalar('Accuracy', train_acc[epoch] / 100.0, epoch)
        valid_writer.add_scalar('Loss', valid_loss[epoch], epoch)
        valid_writer.add_scalar('Accuracy', valid_acc[epoch] / 100.0, epoch)
        print(f'[{epoch+1:3d}/{epochs:3d}]: ' +
              f'Train Loss {train_loss[epoch]:6.4f}, Acc {train_acc[epoch]:4.1f}% <-> ' +
              f'Valid Loss {valid_loss[epoch]:6.4f}, Acc {valid_acc[epoch]:4.1f}%')
        if scheduler.get_last_lr()[-1] < min_lr:
            min_lr = scheduler.get_last_lr()[-1]
            print(f'Adjusting learning rate --> {min_lr:.1E}')

        if valid_acc[epoch] > best_acc:
            best_acc = valid_acc[epoch]
            torch.save(model.state_dict(), f'./weights/{target_model}_{run_name}_{epoch + 1}Epoch_{int(best_acc)}.pth')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--gpu", help= "gpu id", default= 0, type= int)
    parser.add_argument("-lr", help= "Learning rate", default= 1e-3, type= float)
    parser.add_argument("-ep", help= "Epochs", default= 200, type= int)
    parser.add_argument("-eval", action= "store_true", default= False)
    parser.add_argument("-n", "--name", help= "Experiment name", default= "", type= str)
    parser.add_argument("-m", "--model", help= "target_model", default= 'VGG19', choices= ['VGG19', 'ResNet50'], type= str)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if not args.eval:
        train_dataset = ButterflyMothLoader('./dataset', 'train')
        valid_dataset = ButterflyMothLoader('./dataset', 'valid')
        batch_size = 32 if args.model == 'VGG19' else 64
        lr = args.lr
        epochs = args.ep
        train(args.model, train_dataset, valid_dataset, epochs, lr, batch_size, device, args.name)
    else:
        train_dataset = ButterflyMothLoader('./dataset', 'train')
        test_dataset = ButterflyMothLoader('./dataset', 'test')
        model = VGG19().to(device)
        model.load_state_dict(torch.load('./weights/VGG19_Adam_bn_DataAug2_3_137Epoch_92.pth'))
        train_acc = test(model, train_dataset, 32)
        test_acc = test(model, test_dataset, 32)
        print(f'VGG19:   \tTrain Acc {train_acc:4.1f}% | Test Acc {test_acc:4.1f}%')

        model = ResNet50().to(device)
        model.load_state_dict(torch.load('./weights/ResNet50_Adam_dataAug2_3_127Epoch_95.pth'))
        train_acc = test(model, train_dataset, 64)
        test_acc = test(model, test_dataset, 64)
        print(f'ResNet50:\tTrain Acc {train_acc:4.1f}% | Test Acc {test_acc:4.1f}%')
