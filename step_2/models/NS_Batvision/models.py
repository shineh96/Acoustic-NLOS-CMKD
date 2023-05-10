import torch.nn as nn
from models.NS_Batvision.networks import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder = SpectrogramEncoder()
        self.generator = UNet(opt.image_size)

    def forward(self, x):
        x = x.reshape(-1,64,256,512)
        x = self.encoder(x)
        x = self.generator(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.discriminator = NLayerDiscriminator(opt.image_size)
        
    def forward(self, x):
        x = self.discriminator(x)

        return x
