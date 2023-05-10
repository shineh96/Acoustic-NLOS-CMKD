import torch 
import torch.nn as nn
from models.NS_3DCNN.networks import *


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.encoder = Encoder3D(opt.image_size)
        self.generator = Att_UNet(opt.image_size, opt.grayscale)
                     
        
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 4, 1)) # 3d
        x = self.encoder(x)

        x, e1, e2, e3, e4 = self.generator(x)

        return x, e1, e2, e3, e4


class t_Generator(nn.Module):
    def __init__(self, opt):
        super(t_Generator, self).__init__()
        self.generator = Att_UNet(opt.image_size, opt.grayscale)

    def forward(self, x):
        x, e1, e2, e3, e4 = self.generator(x)

        return x, e1, e2, e3, e4

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.discriminator = NDiscriminator(opt.grayscale)
        
    def forward(self, x):
        x = self.discriminator(x)

        return x
