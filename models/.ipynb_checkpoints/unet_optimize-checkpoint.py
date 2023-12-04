import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models.segmentation as seg
import numpy as np

from models.architecture import double_conv, DoubleConv, DoubleConv2 , SeparableConv2d, DoubleSepConv, SEblock, Res2Net, up_conv, DANet, Attblock , PAM, CAM, SANet
from models.architecture import RGBIR_RDN, RGB2HSV, ASPP_module, _make_layer, _make_MG_unit, ViT, Att_rgbih
##### transfomer ###
from models.architecture import ConvAttention, PreNorm, FeedForward, Res2Attention, MLP, MSAttention
from models.adach import AdaCh

class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        # self.adach = AdaCh

    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
    
class HUNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()
        
        self.h = RGB2HSV()
        self.adach = AdaCh()
        # self.gt = gt
        self.weight_rgbi = nn.Parameter(torch.ones(1,128,1,1))
        self.weight_h = nn.Parameter(torch.ones(1,128,1,1))

        # self.down1 = nn.Conv2d(-1, 64)
        self.dconv_down1 = DoubleConv(n_channels, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)
        
        self.dconv_down4 = DoubleConv(256, 512)
        
        self.dconv_downh1 = DoubleConv(1, 32)
        self.dconv_downh2 = DoubleConv(32, 64)
        self.dconv_downh3 = DoubleConv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.att1 = Att_rgbih(32)
        self.att2 = Att_rgbih(64)
        self.att3 = Att_rgbih(128)


        self.dconv_up3 = DoubleConv(128 + 512, 256)
        self.dconv_up2 = DoubleConv(64 + 256, 128)
        self.dconv_up1 = DoubleConv(32 + 128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, x):
        # x = torch.cat([x, hue], dim=1)
        hue = self.h(x)
        
#         ir_w = (x[:,3,:,:].unsqueeze(1)*self.weight_ir)
#         x = torch.cat((x[:,:3,:,:],ir_w),dim=1)
#         h_w = hue*self.weight_h
        
        x1 = self.dconv_down1(x)
        x = self.maxpool(x1)
        x2 = self.dconv_down2(x)
        x = self.maxpool(x2)
        x3 = self.dconv_down3(x)
        x = self.maxpool(x3)
        
        h1 = self.dconv_downh1(hue)
        hue = self.maxpool(h1)
        h2 = self.dconv_downh2(hue)
        hue = self.maxpool(h2)
        h3 = self.dconv_downh3(hue)
        hue = self.maxpool(h3)
     
        a1 = self.att1(x1, h1)
        a2 = self.att2(x2, h2)
        a3 = self.att3(x3, h3)
        
        rgbih = torch.cat((x*self.weight_rgbi, hue*self.weight_h),dim=1)
        x = self.dconv_down4(rgbih)
        
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, a3], dim=1)
        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, a2], dim=1)
        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, a1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out      

from .full_u_net import *
class FullUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FullUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = double_conv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    
class UNet_dropout(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.dropout = nn.Dropout2d()

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dropout(self.dconv_down1(x))
        x = self.maxpool(conv1)

        conv2 = self.dropout(self.dconv_down2(x))
        x = self.maxpool(conv2)

        conv3 = self.dropout(self.dconv_down3(x))
        x = self.maxpool(conv3)

        x = self.dropout(self.dconv_down4(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dropout(self.dconv_up3(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dropout(self.dconv_up2(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dropout(self.dconv_up1(x))

        out = self.conv_last(x)

        return out
    
#####    
class UNET2(nn.Module):
    def __init__(
            self, in_channels=4, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET2, self).__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_path_prob = 0.0
        # 就是网络模型的某个参数复制分配到不同的GPU的时候，部分参数始终在GPU_0上

        # down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels=feature

        #up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        skip_connections = []

        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)

        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x=TF.resize(x, size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection, x), dim=1)
            x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    




###    
class Res2_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(Res2_UNET, self).__init__()
         
        self.inc = DoubleConv(in_channels, 64)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.ups=nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)
        x = self.pool(x) 
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)
        
        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x=TF.resize(x, size=skip_connection.shape[2:])
                # x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection, x), dim=1)
            x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class AttUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttUNET, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=256, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=128, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=64, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.pool(x1) 
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
        
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.att3(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.att2(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.att1(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.att0(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
    
        return fx   
    
class Res2_AttUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1):
        super(Res2_AttUNET, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=256, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=128, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=64, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.pool(x1) 
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
       
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.att3(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.att2(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.att1(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.att0(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
    
        return fx    
    
class DAUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1):
        super(DAUNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.de4 = up_conv(1024, 512)
        self.da4= DANet(512, 512, 256)   
        self.up4 = DoubleConv(1024, 512)
        
        self.de3 = up_conv(512, 256)
        self.da3= DANet(256, 256, 128)   
        self.up3 = DoubleConv(512, 256)
        
        self.de2 = up_conv(256, 128)
        self.da2= DANet(128, 128, 64)   
        self.up2 = DoubleConv(256, 128)
        
        self.de1 = up_conv(128, 64)
        self.da1= DANet(64, 64, 32)   
        self.up1 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.pool(x1) 
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
        
        d4 = self.de4(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.da4(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up4(d4)
      
        d3 = self.de3(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.da3(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up3(d3)
        
        d2 = self.de2(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.da2(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up2(d2)
        
        d1 = self.de1(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.da1(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up1(d1)
        
        fx = self.final_conv(d1)
    
        return fx
    
class Res2_DAUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1):
        super(Res2_DAUNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Res2Net(64, 128)
        self.down2 = Res2Net(128, 256)
        self.down3 = Res2Net(256, 512)
      
        self.de3 = up_conv(1024, 512)
        self.da3= DANet(512, 512)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.da2= DANet(256, 256)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.da1= DANet(128, 128)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.da0= DANet(64, 64)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.pool(x1) 
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        x5=self.bottleneck(x5)
       
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.da3(x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.da2(x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d2, size=x2.shape[2:])
        t2 = self.da1(x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x1.shape[2:])
        t1 = self.da0(x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
        return fx
    
class Res2_SAUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1):
        super(Res2_SAUNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = Res2Net(64, 128)
        self.down3 = Res2Net(128, 256)
        self.down4 = Res2Net(256, 512)

        self.de4 = up_conv(1024, 512)
        self.da4= SANet(512, 512, 512)   
        self.up4 = DoubleConv(1024, 512)
        
        self.de3 = up_conv(512, 256)
        self.da3= SANet(256, 256, 256)   
        self.up3 = DoubleConv(512, 256)
        
        self.de2 = up_conv(256, 128)
        self.da2= SANet(128, 128, 128)   
        self.up2 = DoubleConv(256, 128)
        
        self.de1 = up_conv(128, 64)
        self.da1= SANet(64, 64, 64)   
        self.up1 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.pool(x1) 
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
        
        d4 = self.de4(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.da4(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up4(d4)
      
        d3 = self.de3(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.da3(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up3(d3)
        
        d2 = self.de2(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.da2(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up2(d2)
        
        d1 = self.de1(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.da1(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up1(d1)
        
        fx = self.final_conv(d1)
    
        return fx
    
    
class RDN_Res2_AttUNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDN_Res2_AttUNET, self).__init__()
        
        self.inrdn = RGBIR_RDN(in_channels)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=256, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=128, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=64, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        x1 = self.inrdn(x)
        x2 = self.pool(x1) 
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
       
        d4 = self.de3(x5)
        if d4.shape != x4.shape:
                 d4=TF.resize(d4, size=x4.shape[2:])
        t4 = self.att3(d4, x4)
        d4 = torch.cat((t4, d4), dim=1)
        d4 = self.up3(d4)
      
        d3 = self.de2(d4)
        if d3.shape != x3.shape:
                 d3=TF.resize(d3, size=x3.shape[2:])
        t3 = self.att2(g=d3, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)
        
        d2 = self.de1(d3)
        if d2.shape != x2.shape:
                 d2=TF.resize(d4, size=x2.shape[2:])
        t2 = self.att1(g=d2, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)
        
        d1 = self.de0(d2)
        if d1.shape != x1.shape:
                 d1=TF.resize(d1, size=x4.shape[2:])
        t1 = self.att0(g=d1, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        
        fx = self.final_conv(d1)
    
        return fx
       
class Res2_AttUNET_Sup(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):   #features=[64, 128, 256, 512]
        super(Res2_AttUNET_Sup, self).__init__()
        self.h = RGB2HSV()
        self.inc = DoubleConv(2, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Res2Net(64, 128)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = Res2Net(128, 256)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = Res2Net(256, 512)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.de3 = up_conv(1024, 512)
        self.att3= Attblock(F_g=512, F_l=512, F_int=256)   
        self.up3 = DoubleConv(1024, 512)
        
        self.de2 = up_conv(512, 256)
        self.att2= Attblock(F_g=512, F_l=256, F_int=128)   
        self.up2 = DoubleConv(512, 256)
        
        self.de1 = up_conv(256, 128)
        self.att1= Attblock(F_g=512, F_l=128, F_int=64)   
        self.up1 = DoubleConv(256, 128)
        
        self.de0 = up_conv(128, 64)
        self.att0= Attblock(F_g=512, F_l=64, F_int=32)   
        self.up0 = DoubleConv(128, 64)

        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        

        self.fn4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.fn3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.fn2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.fn1 = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        hue = self.h(x)
        x = torch.cat([x[:,3:,:,:], hue], dim=1)
        x1 = self.inc(x)
        x2 = self.pool(x1) 
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        
        x5=self.bottleneck(x5)
        logits = []
        
        g = self.de3(x5)   # (1, 512, 35, 35)
        if g.shape != x4.shape:
                 g=TF.resize(g, size=x4.shape[2:])
        t4 = self.att3(g, x4)
        d4 = torch.cat((t4, g), dim=1)
        d4 = self.up3(d4)  # (1, 512, 64, 64)
        p4 = self.fn4(d4)
        p4 = self.upscore4(p4)
        p4 = torch.sigmoid(p4)
        logits.append(p4)
      
        d3 = self.de2(d4)  # (1, 256, 143, 143)
        if g.shape != x3.shape:
                 g=TF.resize(g, size=x3.shape[2:])
        t3 = self.att2(g=g, x=x3)
        d3 = torch.cat((t3, d3), dim=1)
        d3 = self.up2(d3)  # (1, 256, 128, 128)
        p3 = self.fn3(d3)
        p3 = self.upscore3(p3)
        p3 = torch.sigmoid(p3)
        logits.append(p3)

        
        d2 = self.de1(d3)
        if g.shape != x2.shape:
                 g=TF.resize(g, size=x2.shape[2:])
        t2 = self.att1(g=g, x=x2)
        d2 = torch.cat((t2, d2), dim=1)
        d2 = self.up1(d2)   # (1, 128)
        p2 = self.fn2(d2)
        p2 = self.upscore2(p2)
        p2 = torch.sigmoid(p2)

        logits.append(p2)
        
        d1 = self.de0(d2)
        if g.shape != x1.shape:
                 g=TF.resize(g, size=x1.shape[2:])
        t1 = self.att0(g=g, x=x1)
        d1 = torch.cat((t1, d1), dim=1)
        d1 = self.up0(d1)
        p1 = self.fn1(d1)  
        p1 = torch.sigmoid(p1)
        logits.append(p1)
    
        return logits
        
    
class Simple_Res2Unet(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256],
    ):
        super(Simple_Res2Unet, self).__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_path_prob = 0.0
        # 就是网络模型的某个参数复制分配到不同的GPU的时候，部分参数始终在GPU_0上

        # down part
        for feature in features:
            self.downs.append(Res2Net(in_channels,feature))
            in_channels=feature

        #up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Res2Net(feature*2, feature))

        self.bottleneck = Res2Net(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)

        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1] # 反向操做

        for idx in range(0, len(self.ups), 2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x=TF.resize(x, size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection, x), dim=1)
            x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    


###### TransUNet #####
## https://github.com/mkara44/transunet_pytorch/blob/main/utils/vit.py
from einops import rearrange
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        if x_concat is not None:
            # print("x_cat",x_concat.shape )
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, head_num, mlp_dim, block_num, patch_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_size = img_size // patch_size
        self.vit = ViT(out_channels * 8, out_channels * 8, img_size, head_num, mlp_dim, block_num, patch_size)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.vit(x)
        # x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_size, y=self.vit_img_size)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        # self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        # self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        # self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        # self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.decoder1 = DecoderBottleneck(out_channels * 12, out_channels*4)
        self.decoder2 = DecoderBottleneck(out_channels * 6, out_channels*2)
        self.decoder3 = DecoderBottleneck(out_channels * 3, out_channels)
        self.decoder4 = DecoderBottleneck(out_channels, int(out_channels * 1/2))

        self.conv1 = nn.Conv2d(int(out_channels * 1/2), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    def __init__(self, in_channels,class_num, img_size=16, out_channels=64, head_num=4, mlp_dim=1024, block_num=6, patch_dim=1):
        super().__init__()

        self.encoder = Encoder(in_channels, out_channels, img_size, head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        return x

############ cvt #########
import gc
from einops.layers.torch import Rearrange

    
class TransformerBlock(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.sz = img_size
        self.mha = Res2Attention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.mlp = SEb(dim, mlp_dim, dropout=dropout)
        self.mlp = MLP(dim, mlp_dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        mha = self.mha(x)
        ln1 = self.ln1(mha)
        mha = self.dropout(ln1)
        x = mha + x + h
        # x = rearrange(x, 'b (h w) c -> b c h w', h=self.sz  ,w=self.sz ) 
        # print('x',x.shape)
        mlp = self.mlp(x)
        ln2 = self.ln2(mlp)
        x = ln2 + x
        return x


class Trans_depth(nn.Module):  # mlp -> SE
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.sz = img_size
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.1) for _ in range(depth)])

    def forward(self, x):
        x, h = torch.chunk(x, 2, dim = 2)
        for block in self.layers:
            x = block(x, h)
        return x
    
class CvTblock(nn.Module):
    def __init__(self, in_ch, out_ch,  image_size, kernels, strides, padding, 
                 patch, heads , depth, dropout=0.1, emb_dropout=0., scale_dim=4):
        super().__init__()

        self.dim = 64
        self.head = heads
        self.depth = depth
        self.img_sz_patch = image_size//patch

        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernels, strides, padding),
            Rearrange('b c h w -> b (h w) c', h = self.img_sz_patch, w = self.img_sz_patch),
            nn.LayerNorm(out_ch)
        )
        self.hue_embed = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernels, strides, padding),
            Rearrange('b c h w -> b (h w) c', h = self.img_sz_patch, w = self.img_sz_patch),
            nn.LayerNorm(out_ch)
        )
        self.transformer = nn.Sequential(
            Trans_depth(out_ch, image_size//patch, self.depth, self.head, dim_head=self.dim,
                                              mlp_dim=out_ch * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = self.img_sz_patch, w = self.img_sz_patch)
        )

    def forward(self, img, hue):
        x = self.conv_embed(img)  # ([1, 4096, 64])
        h = self.hue_embed(hue)   # ([1, 4096, 1])

        xh = torch.cat([x,h], dim = 2)
        # print('xh', xh.shape, type(xh))
        x = self.transformer(xh)
        h = rearrange(h, 'b (h w) c -> b c h w', h=self.img_sz_patch, w=self.img_sz_patch)
        return x, h
    
class Res2VTUnet(nn.Module):
    def __init__(self, in_ch, n_class, dim = 32, image_size=256, kernels=[5, 3, 3], strides=[4, 2, 2], padding=[1, 1, 1],
                 patch=[4,8,16], heads=[2, 2, 4] , depth = [2, 2, 4], dropout=0.1, emb_dropout=0.):
        super().__init__()
        self.h = RGB2HSV()
        self.h_conv = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)
        self.in_conv = nn.Conv2d(in_ch, dim, kernel_size=3, stride=1, padding=1)
                     
        self.w1_rgbi = nn.Parameter(torch.ones(1,dim,1,1))
        self.w1_h = nn.Parameter(torch.ones(1,dim,1,1))
        self.w2_rgbi = nn.Parameter(torch.ones(1,dim*2,1,1))
        self.w2_h = nn.Parameter(torch.ones(1,dim*2,1,1))
        self.w3_rgbi = nn.Parameter(torch.ones(1,dim*4,1,1))
        self.w3_h = nn.Parameter(torch.ones(1,dim*4,1,1))


        self.cvt1 = CvTblock(dim, dim*2, image_size, kernels=kernels[0], strides=strides[0], 
                             padding=padding[0], patch=patch[0], heads=heads[0] , depth = depth[0])
        self.cvt2 = CvTblock(dim*2, dim*4, image_size, kernels=kernels[1], strides=strides[1], 
                             padding=padding[1], patch=patch[1],heads=heads[1] , depth = depth[1])
        self.cvt3 = CvTblock(dim*4, dim*8, image_size, kernels=kernels[2], strides=strides[2], 
                             padding=padding[2], patch=patch[2],heads=heads[2] , depth = depth[2])
        
        self.dconv_up3 = DoubleConv(dim*4 + dim*8, dim*4)
        self.dconv_up2 = DoubleConv(dim*2 + dim*4, dim*2)
        self.dconv_up1 = DoubleConv(dim + dim*2, dim)

        self.conv_last = nn.Conv2d(dim, n_class, 1)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):    
        hue = self.h(x)
        h = self.h_conv(hue)
        x1=self.in_conv(x)
        # print('c1', x1.shape)
        x2, h2 = self.cvt1(x1*self.w1_rgbi, h*self.w1_h)  # torch.Size([1, 64, 64, 64])
        # print('c2', x2.shape, h2.shape)
        x3, h3 = self.cvt2(x2*self.w2_rgbi, h2*self.w2_h)
        # print('c3', x3.shape, h3.shape)
        
        bt, _ = self.cvt3(x3*self.w3_rgbi, h3*self.w3_h)
        # print('bt', x.shape)
  
        x = nn.functional.interpolate(bt, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        d3 = self.dconv_up3(x)
        # print('d3', x.shape)
        
        x = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        d2 = self.dconv_up2(x)
        # print('d2', x.shape)
        
        x = nn.functional.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.dconv_up1(x)        
        x = self.dropout(x)
        out = self.conv_last(x)
        # print('o', out.shape)
        return out

    
    
############  ablation  #############
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.sz = img_size
        self.mha = ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.mlp = SEb(dim, mlp_dim, dropout=dropout)
        self.mlp = MLP(dim, mlp_dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        mha = self.mha(x)
        ln1 = self.ln1(mha)
        mha = self.dropout(ln1)
        x = mha + x + h
        # x = rearrange(x, 'b (h w) c -> b c h w', h=self.sz  ,w=self.sz ) 
        # print('x',x.shape)
        mlp = self.mlp(x)
        ln2 = self.ln2(mlp)
        x = ln2 + x
        return x


class Trans_depth(nn.Module):  # mlp -> SE
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.sz = img_size
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.1) for _ in range(depth)])

    def forward(self, x):
        # print('db', x.shape)
        # print('db_h', h.shape)
        # x = x[:,:,:-1]
        # h = x[:,:,-1].unsqueeze(2)
        x, h = torch.chunk(x, 2, dim = 2)
        for block in self.layers:
            x = block(x, h)
        return x
    
class CvTblock(nn.Module):
    def __init__(self, in_ch, out_ch,  image_size, kernels, strides, padding, 
                 patch, heads , depth, dropout=0.1, emb_dropout=0., scale_dim=4):
        super().__init__()

        self.dim = 64
        self.head = heads
        self.depth = depth
        self.img_sz_patch = image_size//patch

        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernels, strides, padding),
            Rearrange('b c h w -> b (h w) c', h = self.img_sz_patch, w = self.img_sz_patch),
            nn.LayerNorm(out_ch)
        )
        self.hue_embed = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernels, strides, padding),
            Rearrange('b c h w -> b (h w) c', h = self.img_sz_patch, w = self.img_sz_patch),
            nn.LayerNorm(out_ch)
        )
        self.transformer = nn.Sequential(
            Trans_depth(out_ch, image_size//patch, self.depth, self.head, dim_head=self.dim,
                                              mlp_dim=out_ch * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = self.img_sz_patch, w = self.img_sz_patch)
        )

    def forward(self, img, hue):
        x = self.conv_embed(img)  # ([1, 4096, 64])
        h = self.hue_embed(hue)   # ([1, 4096, 1])

        xh = torch.cat([x,h], dim = 2)
        # print('xh', xh.shape, type(xh))
        x = self.transformer(xh)
        h = rearrange(h, 'b (h w) c -> b c h w', h=self.img_sz_patch, w=self.img_sz_patch)
        return x, h
    
class HueVTUnet(nn.Module):
    def __init__(self, in_ch, n_class, dim = 32, image_size=256, kernels=[5, 3, 3], strides=[4, 2, 2], padding=[1, 1, 1],
                 patch=[4,8,16], heads=[2, 2, 4] , depth = [2, 2, 4], dropout=0.1, emb_dropout=0.):
        super().__init__()
        self.h = RGB2HSV()
        self.h_conv = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)
        self.in_conv = nn.Conv2d(in_ch, dim, kernel_size=3, stride=1, padding=1)
                     
        self.w1_rgbi = nn.Parameter(torch.ones(1,dim,1,1))
        self.w1_h = nn.Parameter(torch.ones(1,dim,1,1))
        self.w2_rgbi = nn.Parameter(torch.ones(1,dim*2,1,1))
        self.w2_h = nn.Parameter(torch.ones(1,dim*2,1,1))
        self.w3_rgbi = nn.Parameter(torch.ones(1,dim*4,1,1))
        self.w3_h = nn.Parameter(torch.ones(1,dim*4,1,1))


        self.cvt1 = CvTblock(dim, dim*2, image_size, kernels=kernels[0], strides=strides[0], 
                             padding=padding[0], patch=patch[0], heads=heads[0] , depth = depth[0])
        self.cvt2 = CvTblock(dim*2, dim*4, image_size, kernels=kernels[1], strides=strides[1], 
                             padding=padding[1], patch=patch[1],heads=heads[1] , depth = depth[1])
        self.cvt3 = CvTblock(dim*4, dim*8, image_size, kernels=kernels[2], strides=strides[2], 
                             padding=padding[2], patch=patch[2],heads=heads[2] , depth = depth[2])
        
        self.dconv_up3 = DoubleConv(dim*4 + dim*8, dim*4)
        self.dconv_up2 = DoubleConv(dim*2 + dim*4, dim*2)
        self.dconv_up1 = DoubleConv(dim + dim*2, dim)

        self.conv_last = nn.Conv2d(dim, n_class, 1)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):    
        hue = self.h(x)
        h = self.h_conv(hue)
        x1=self.in_conv(x)
        # print('c1', x1.shape)
        x2, h2 = self.cvt1(x1*self.w1_rgbi, h*self.w1_h)  # torch.Size([1, 64, 64, 64])
        # print('c2', x2.shape, h2.shape)
        x3, h3 = self.cvt2(x2*self.w2_rgbi, h2*self.w2_h)
        # print('c3', x3.shape, h3.shape)
        
        bt, _ = self.cvt3(x3*self.w3_rgbi, h3*self.w3_h)
        # print('bt', x.shape)
  
        x = nn.functional.interpolate(bt, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        d3 = self.dconv_up3(x)
        # print('d3', x.shape)
        
        x = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        d2 = self.dconv_up2(x)
        # print('d2', x.shape)
        
        x = nn.functional.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.dconv_up1(x)        
        x = self.dropout(x)
        out = self.conv_last(x)
        # print('o', out.shape)
        return out