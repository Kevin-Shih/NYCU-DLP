import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import ToTensor, ToPILImage
import os
import numpy as np

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./dataset/train.csv')
    elif mode == 'valid':
        df = pd.read_csv('./dataset/valid.csv')
    else:
        df = pd.read_csv('./dataset/test.csv')
    path = df['filepaths'].tolist()
    label = df['label_id'].tolist()
    return path, label

class ButterflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d %s images..." % (len(self.img_name), mode))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root, self.img_name[index])
        img = Image.open(path)
        if self.mode == 'train':
            rnd = np.random.random()
            if rnd < 0.15:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif rnd < 0.3:
                img = img.transpose(Image.Transpose.ROTATE_90)
            elif rnd < 0.45:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif rnd < 0.6:
                img = img.transpose(Image.Transpose.ROTATE_270)

            rnd = np.random.random() * 0.5
            rnd_x = np.random.random() * rnd
            rnd_y = np.random.random() * rnd
            l_bound = int(rnd_x * 224)
            r_bound = int((1 - rnd + rnd_x) * 224)
            u_bound = int(rnd_y * 224)
            d_bound = int((1 - rnd + rnd_y) * 224)
            img = img.crop((l_bound, u_bound, r_bound, d_bound)).resize((224, 224))
        return ToTensor()(img), self.label[index]

if __name__ == '__main__':
    a = ButterflyMothLoader('./dataset', 'train')
    img, label = a.__getitem__(0)
    print(img.shape, label)
    img = ToPILImage()(img)
    img.show()
    