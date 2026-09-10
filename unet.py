import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self,in_channels=1):
        super().__init__()

        self.down1=DoubleConv(in_channels,64)
        self.down2=DoubleConv(64,128)
        self.down3=DoubleConv(128,256)
        self.down4=DoubleConv(256,512)
        self.bottleneck=DoubleConv(512,1024)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        skip1=self.down1(x)
        x=self.pool(skip1)
        skip2=self.down2(x)
        x=self.pool(skip2)
        skip3 = self.down3(x)
        x = self.pool(skip3)
        skip4 = self.down4(x)
        x = self.pool(skip4)
        x=self.bottleneck(x)
        return x, skip1, skip2, skip3, skip4

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.conv=DoubleConv(in_channels,out_channels)

    def forward(self,x,skip):
        x=self.up(x)
        x=torch.cat([x,skip],dim=1)
        x=self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_block1=DecoderBlock(1024,512)
        self.decoder_block2=DecoderBlock(512,256)
        self.decoder_block3=DecoderBlock(256,128)
        self.decoder_block4=DecoderBlock(128,64)
        self.final_conv=nn.Conv2d(64,1,kernel_size=1)

    def forward(self,x,skip4,skip3,skip2,skip1):
        x=self.decoder_block1(x,skip4)
        x=self.decoder_block2(x,skip3)
        x=self.decoder_block3(x,skip2)
        x=self.decoder_block4(x,skip1)
        x=self.final_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self,x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x=self.decoder(x,skip4,skip3,skip2,skip1)
        return x

# model=UNet()
# x=torch.randn(1,1,128,128)
# out=model(x)
# print(x.shape)
# print(out.shape)