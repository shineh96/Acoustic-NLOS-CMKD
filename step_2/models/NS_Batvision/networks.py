import torch.nn.functional as F
import torch.nn as nn
import torch

class encode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(encode_block, self).__init__()
        self.double = double
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        if self.double: # Quick bugfix without testing. DDP will not work if parameters are defined which are not used in the computation.
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
                )

    def forward(self, x):
        x = self.down_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x

class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super(SpectrogramEncoder, self).__init__()

        self.enc1  = encode_block(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=(2,1))            
        self.enc2  = encode_block(64, 128, 3, 1, (2,1), False)
        self.enc3  = encode_block(128, 128, 3, 1, (2,1), False)    
        self.enc4  = encode_block(128, 256, 3, 1, 2, False)
        self.enc5  = encode_block(256, 512, 3, 1, 2, False)
        self.enc6  = encode_block(512, 512, 3, 1, 2, False)
        self.enc7  = encode_block(512, 1024, 3, 1, 2, False)
        self.enc8  = encode_block(1024, 1024, (3,4), 1, 2, False)
        
        self.fc1   = fc(16384,1024)
        self.fc2   = fc(1024,1024)

    def forward(self, x):
        
        x = self.enc1(x)  
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class UNet(nn.Module):
    def __init__(self, output=64):
        super(UNet, self).__init__()

        self.enc1_0 = encode_block(1, 64, 3, 1, 1, False)
        self.enc1_1 = encode_block(64, 64, 3, 1, 1, False)
        self.epool1 = nn.MaxPool2d(2, 2)
        
        self.enc2_0 = encode_block(64, 128, 3, 1, 1, False)
        self.epool2 = nn.MaxPool2d(2, 2)
        
        self.enc3_0 = encode_block(128, 256, 3, 1, 1, False)
        self.epool3 = nn.MaxPool2d(2, 2)
        
        self.enc4_0 = encode_block(256, 256, 3, 1, 1, False)
        self.epool4 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = encode_block(256, 512, 3, 1, 1, False)
        
        self.dec4_0 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec4_1 = encode_block(512, 256, 3, 1, 1, False)
        
        self.dec3_0 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.dec3_1 = encode_block(512, 256, 3, 1, 1, False)
        
        self.dec2_0 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2_1 = encode_block(256, 128, 3, 1, 1, False)
        
        self.dec1_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1_1 = encode_block(128, 128, 3, 1, 1, False)

        self.dec0_0 = nn.ConvTranspose2d(128, 64, 4 if output >= 64 else 3, 2 if output >= 64 else 1, 1)
        self.dec0_1 = encode_block(64, 64, 3, 1, 1, False)

        self.dec00_0 = nn.ConvTranspose2d(64, 64, 4 if output == 128 else 3, 2 if output==128 else 1, 1)
        self.dec00_1 = encode_block(64, 64, 3, 1, 1, False)

        self.final  = nn.Conv2d(64, 1, 1)  
        # output 128x128x1

    def forward(self, x):        
        x = x.view(-1, 1, 32, 32)

        x = self.enc1_0(x)
        x1 = self.enc1_1(x)
        x = self.epool1(x1)
        
        x2 = self.enc2_0(x)
        x = self.epool2(x2)
        
        x3 = self.enc3_0(x)
        x = self.epool3(x3)
        
        x4 = self.enc4_0(x)
        x = self.epool4(x4)
        
        x = self.bottleneck(x)
        
        x = torch.cat([self.dec4_0(x),x4], 1)
        x = self.dec4_1(x)
        
        x = torch.cat([self.dec3_0(x),x3], 1)
        x = self.dec3_1(x)
        
        x = torch.cat([self.dec2_0(x),x2], 1)
        x = self.dec2_1(x)
        
        x = torch.cat([self.dec1_0(x),x1], 1)
        x = self.dec1_1(x)

        x = self.dec0_0(x)
        x = self.dec0_1(x)

        x = self.dec00_0(x)
        x = self.dec00_1(x)

        x = self.final(x)
        return x
        
class fc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fc, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.fc(x)
        
        return x

class NLayerDiscriminator(nn.Module):
    def __init__(self,output=64):
        super().__init__()
        self.output = output

        if output == 128:
            self.final = 256
            self.k = 4
            self.s = [2,2,2]
        elif output == 64:
            self.final = 128
            self.k = 4
            self.s = [1,1,2]
        elif output == 32:
            self.final = 128
            self.k = 3
            self.s = [1,2,1]

        self.conv1 = nn.Conv2d(1, 64, self.k, self.s[0], 1)
        self.act1  = nn.LeakyReLU(0.2, True)

        self.conv2 = nn.Conv2d(64, 128, self.k, self.s[1], 1)
        self.bn2   = nn.BatchNorm2d(128)
        self.act2  = nn.LeakyReLU(0.2, True)

        if output == 128:
            self.conv3 = nn.Conv2d(128, 256, self.k, self.s[2], 1)
            self.bn3   = nn.BatchNorm2d(256)
            self.act3  = nn.LeakyReLU(0.2, True)

        self.final = nn.Conv2d(self.final, 1, self.k, self.s[2], 1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.output == 128:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.act3(x)
        x = self.final(x)

        return x