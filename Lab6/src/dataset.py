import os
import json
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def custom_transform(mode):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if mode == 'train':
        transformer = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    return transformer

def get_data(mode, root):
    label_map = json.load(open('objects.json'))
    if mode == 'train':
        data_json:dict = json.load(open('train.json'))
        img_path = list(data_json.keys())
        labels_list = list(data_json.values())
    else:
        img_path = None
        path = 'new_test.json' if mode == 'new_test' else 'test.json'
        data_json = json.load(open(path))
        labels_list = list(data_json)

    one_hot_list = []
    # len(labels_list)
    for labels in labels_list:
        one_hot = np.zeros(24, dtype=np.int8)
        for label in labels:
            one_hot[label_map[label]] = 1
        one_hot_list.append(one_hot)
    return img_path, one_hot_list

class IclevrDataset(Dataset):
    def __init__(self, mode, root):
        self.mode = mode
        self.root = root
        self.img_path, self.labels = get_data(mode, root)
        self.transform = custom_transform(mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.root, self.img_path[idx])).convert('RGB')
            image = self.transform(image)
            label = Tensor(self.labels[idx])
            return image, label
        else:
            label = Tensor(self.labels[idx])
            return label


if __name__ == '__main__':
    dataset = IclevrDataset(mode='train', root='iclevr')
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(train_dataloader), len(dataset))
    for i, label in enumerate(tqdm(train_dataloader)):
        pass