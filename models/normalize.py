import torch
import torch.nn as nn

class Normalize(nn.Module):
    '''A layer for normalizing input before passing it into the network'''
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

def apply_normalization(base_model, mean, std):
    return nn.Sequential(Normalize(mean=mean, std=std), base_model)