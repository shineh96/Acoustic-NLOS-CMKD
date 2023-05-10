import torch 
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, double=False):
        super(conv_block, self).__init__()
        
        self.double = double
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True) 
        )
        if self.double:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True) 
            )
    
    def forward(self, x):
        x = self.down_conv(x)
        if self.double:
            x = self.conv(x)

        return x 

class conv_3d_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, downsample=None):
        super(conv_3d_block, self).__init__() 
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) 
        self.bn1 = nn.BatchNorm3d(outplanes) 
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv3d(outplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) 
        self.bn2 = nn.BatchNorm3d(outplanes) 
        self.downsample = downsample 
        self.stride = stride 

    def forward(self, x):
        residual = x

        out = self.conv1(x) 
        out = self.bn1(out) 
        out = self.relu(out) 

        return out 

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

# https://github.com/LeeJunHyun/Image_Segmentation/
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__() 
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) 
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() 
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g) 
        x1 = self.W_x(x) 
        psi = self.relu(g1 + x1) 
        psi = self.psi(psi) 

        return x * psi 

class Encoder3D(nn.Module):
    def __init__(self, image_size):
        super(Encoder3D, self).__init__() 

        self.enc1 = conv_3d_block(inplanes=8, outplanes=32, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)
        self.enc2 = conv_3d_block(inplanes=32, outplanes=32, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)
        self.enc3 = conv_3d_block(inplanes=32, outplanes=64, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)
        self.enc4 = conv_3d_block(inplanes=64, outplanes=64, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)
        self.enc5 = conv_3d_block(inplanes=64, outplanes=128, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)
        self.enc6 = conv_3d_block(inplanes=128, outplanes=256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.enc7 = conv_3d_block(inplanes=256, outplanes=256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.enc8 = conv_3d_block(inplanes=256, outplanes=512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.fc1 = nn.Linear(1024, 2048)

        self.fc2 = nn.Linear(2048, int(image_size * image_size))

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
        x = nn.ReLU(inplace=True)(x) 
        x = self.fc2(x)
        
        return x 


class Att_UNet(nn.Module):
    def __init__(self, image_size, grayscale):
        super(Att_UNet, self).__init__()
        self.image_size = image_size

        self.enc_pool_1 = nn.MaxPool2d(2, 2) 
        self.enc_1_0 = conv_block(1, 64, 3, 1, 1, True)

        self.enc_pool_2 = nn.MaxPool2d(2,2)
        self.enc_2_0 = conv_block(64, 128, 3, 1, 1, True)

        self.enc_pool_3 = nn.MaxPool2d(2,2)
        self.enc_3_0 = conv_block(128, 256, 3, 1, 1, True)

        self.enc_pool_4 = nn.MaxPool2d(2,2)
        self.enc_4_0 = conv_block(256, 512, 3, 1, 1, True)
        
        self.dec_4_0 = up_conv(512, 256, 3, 1, 1)
        self.dec_4_1 = attention_block(256, 256, 128) 
        self.dec_4_2 = conv_block(512, 256, 3, 1, 1, True)
        
        self.dec_3_0 = up_conv(256, 128, 3, 1, 1)
        self.dec_3_1 = attention_block(128, 128, 64)
        self.dec_3_2 = conv_block(256, 128, 3, 1, 1, True)
        
        self.dec_2_0 = up_conv(128, 64, 3, 1, 1)
        self.dec_2_1 = attention_block(64, 64, 32)
        self.dec_2_2 = conv_block(128, 64, 3, 1, 1, True)

        if grayscale == 'True':
            self.final  = nn.Conv2d(64, 1, 1, 1, 0) 
        else:
            self.final = nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, x):       
        x = x.view(-1, 1, self.image_size, self.image_size)

        enc_1_0 = self.enc_1_0(x)
        enc_1_1 = self.enc_pool_1(enc_1_0)
        
        enc_2_0 = self.enc_2_0(enc_1_1)
        enc_2_1 = self.enc_pool_2(enc_2_0)
        
        enc_3_0 = self.enc_3_0(enc_2_1)
        enc_3_1 = self.enc_pool_3(enc_3_0)

        enc_4_0 = self.enc_4_0(enc_3_1)
        
        dec_4_0 = self.dec_4_0(enc_4_0)
        dec_4_1 = self.dec_4_1(dec_4_0, enc_3_0)
        dec_4_2 = torch.cat([dec_4_0, dec_4_1], dim=1)
        dec_4_2 = self.dec_4_2(dec_4_2)
        
        dec_3_0 = self.dec_3_0(dec_4_2)
        dec_3_1 = self.dec_3_1(dec_3_0, enc_2_0)
        dec_3_2 = torch.cat([dec_3_1, dec_3_0], dim=1)
        dec_3_2 = self.dec_3_2(dec_3_2)
        
        dec_2_0 = self.dec_2_0(dec_3_2)
        dec_2_1 = self.dec_2_1(dec_2_0, enc_1_0)
        dec_2_2 = torch.cat([dec_2_1, dec_2_0], dim=1)
        dec_2_2 = self.dec_2_2(dec_2_2)
 
        final = self.final(dec_2_2)
        
        return final, enc_1_0, enc_2_0, enc_3_0, enc_4_0

class NDiscriminator(nn.Module):
    def __init__(self, grayscale):
        super(NDiscriminator, self).__init__()
        if grayscale == 'True':
            self.conv1 = nn.Conv2d(1, 64, 4, 2, 1) 
        else: 
            self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.bn1   = nn.BatchNorm2d(64)
        self.act1  = nn.LeakyReLU(0.2, True) 

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) 
        self.bn2   = nn.BatchNorm2d(128)
        self.act2  = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) 
        self.bn3   = nn.BatchNorm2d(256)
        self.act3  = nn.LeakyReLU(0.2, True)

        self.final = nn.Conv2d(256, 1, 4, 1, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.final(x)

        return x