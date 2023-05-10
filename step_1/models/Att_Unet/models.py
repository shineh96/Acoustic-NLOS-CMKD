import torch 
import torch.nn as nn
from models.Att_Unet.networks import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.generator = Att_UNet(opt.image_size, opt.grayscale)

    def forward(self, x):
        x = self.generator(x)

        return x