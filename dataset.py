import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os
import numpy as np
import tqdm


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

        if os.path.exists(os.path.join(root_dir, "cache")):
            self.img_cache = torch.load(os.path.join(root_dir, "cache", "img_cache.pt"))
            self.target_cache = torch.load(os.path.join(root_dir, "cache", "target_cache.pt"))
        else:
            os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)
            self.make_cache()
    
    def make_cache(self):
        self.img_cache = torch.tensor([])
        self.target_cache = torch.tensor([])
        for img_path in tqdm.tqdm(self.file_list):
            # TODO: optimize, but also its cached so its fine
            image = torch.as_tensor(np.array(Image.open(img_path).convert("L"), dtype=np.uint8), dtype=torch.float32).unsqueeze(0) / 255.0
            if self.transform:
                image = self.transform(image)
            target = 1 if "NORMAL" in img_path else 0

            self.img_cache = torch.cat((self.img_cache, image), dim=0)
            self.target_cache = torch.cat((self.target_cache, torch.tensor([target])), dim=0)

        torch.save(self.img_cache, os.path.join(self.root, "cache", "img_cache.pt"))
        torch.save(self.target_cache, os.path.join(self.root, "cache", "target_cache.pt"))



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        image = self.img_cache[idx]
        target = self.target_cache[idx]

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