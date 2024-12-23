import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        print(x.size())
        return x
    
class Thrower(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        raise InterruptedError
        return x
        
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class DBlock(nn.Module):
    def __init__(self, start, end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(start, end, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(end, end, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)

class ChestXrayCNNRes(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            DBlock(1,64), # 256x256x64
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x128x64

            DBlock(64,128),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64x128

            DBlock(128,256),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32x256

            DBlock(256,512),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16x512

            DBlock(512,1024),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8x1024

            DBlock(1024,2048),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4x4x2048

            nn.Flatten(),
            nn.Linear(4*4*2048, 2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.block(x)
