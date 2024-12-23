import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os
import numpy as np


IMG_SIZE = 256

# transforms = v2.Compose([
#     #v2.ToDtype(torch.float32, scale=True),
#     v2.PILToTensor(),
#     v2.ToTensor(), 
    
#     v2.Resize(IMG_SIZE), # These two ensure a square center crop.
#     v2.CenterCrop(size=(IMG_SIZE,IMG_SIZE)), 
    

# ])

transforms = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.CenterCrop(size=(IMG_SIZE,IMG_SIZE)),
    v2.Normalize(mean=[0.485], std=[0.229]), # imagenet mean and std
    #v2.ToTensor()
])

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False):
        self.root = root_dir
        self.transform = transform
        self.file_list = glob.glob(os.path.join(root_dir, '*/*.jpeg'), recursive=True)



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        # Its inefficient, but it works. This should be in transforms.
        image = torch.as_tensor(np.array(Image.open(img_path).convert("L"), dtype=np.uint8), dtype=torch.float32).unsqueeze(0) / 255.0
        # print(f"Image type: {type(image)}")
        # print(image.size())
        if self.transform:
            image = self.transform(image)

        target = 1 if "NORMAL" in img_path else 0
        return image, target


def get_train_dataset():
    dataset = ChestXrayDataset(
        root_dir=os.path.expanduser("~/torch_datasets/chest_xray/train"),
        train=True,
        transform=transforms
    )
    return dataset

def get_test_dataset():
    dataset = ChestXrayDataset(
        root_dir=os.path.expanduser("~/torch_datasets/chest_xray/test"),
        train=False,
        transform=transforms
    )
    return dataset

def get_val_dataset():

    dataset = ChestXrayDataset(
        root_dir=os.path.expanduser("~/torch_datasets/chest_xray/val"),
        transform=transforms,
        train=False
    )
    return dataset

def get_dataloader(dataset, batch_size=64):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print('Number of samples: ', len(cap))
    img, target = cap[4] # load 4th sample

    print("Image Size: ", img.size())
    print(target)