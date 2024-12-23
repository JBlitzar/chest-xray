import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os


IMG_SIZE = 256

transforms = v2.Compose([
    #v2.PILToTensor(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(IMG_SIZE), # These two ensure a square center crop.
    v2.CenterCrop(size=(IMG_SIZE,IMG_SIZE)), 
    

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
        image = Image.open(img_path)
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