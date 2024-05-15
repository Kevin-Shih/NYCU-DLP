import os
import torch
import shutil
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        # resize images
        image = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        # trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))
        return ToTensor()(image), ToTensor()(mask)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def augmentation(**sample):
    img = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
    mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
    transpose = None
    rnd = np.random.random()
    if rnd < 0.15:
        transpose = Image.Transpose.FLIP_LEFT_RIGHT
    elif rnd < 0.3:
        transpose = Image.Transpose.ROTATE_90
    elif rnd < 0.45:
        transpose = Image.Transpose.FLIP_TOP_BOTTOM
    elif rnd < 0.6:
        transpose = Image.Transpose.ROTATE_270
    if transpose is not None:
        img = img.transpose(transpose)
        mask = mask.transpose(transpose)

    rnd = np.random.random() * 0.2 + 0.1 # 0.1~0.3
    rnd_x = np.random.random() * rnd
    rnd_y = np.random.random() * rnd
    l_bound = int(rnd_x * 256)
    r_bound = int((1 - rnd + rnd_x) * 256)
    u_bound = int(rnd_y * 256)
    d_bound = int((1 - rnd + rnd_y) * 256)
    img = img.crop((l_bound, u_bound, r_bound, d_bound))
    mask = mask.crop((l_bound, u_bound, r_bound, d_bound))
    sample = dict(image= np.array(img), mask= np.array(mask))
    return sample

def load_dataset(data_path, mode):
    # implement the load dataset function here
    if os.listdir(data_path) == []:
        OxfordPetDataset.download(data_path)
    trans = augmentation if mode == 'train' else None
    return SimpleOxfordPetDataset(data_path, mode= mode, transform= trans)
