import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
import cupy as cp   
import numpy as np   

# from models.BEMD import RGB2BEMD

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SimpleCNN(nn.Module):
    """
    5-layer fully conv CNN
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            double_conv(n_channels, 64),
            double_conv(64, 128),
            nn.Conv2d(128, n_class, 1)
        )
        
    def forward(self, x):
        res = self.conv(x)
        return res
    
class DoubleConv (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class DoubleConv2 (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class DoubleSepConv (nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleSepConv, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

# res2net
class SEblock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
    
class Res2Net(nn.Module):
    expansion = 4  #輸出channel=輸入channel*expansion

    def __init__(self, in_ch, out_ch, downsample=None, stride=1, scales=4, se=False,  norm_layer=None):
        super(Res2Net, self).__init__()
        if out_ch % scales != 0:   # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        
        # bottleneck_planes = groups * planes    
        self.inconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch))
          
        self.x3conv = nn.Sequential(
            nn.Conv2d(out_ch // scales, out_ch // scales,  kernel_size=3, stride=stride, padding=1, groups=out_ch // scales, bias=False),
            nn.Conv2d(out_ch // scales, out_ch // scales, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch // scales))
       
        # self.outconv = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=stride, bias=False),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))
        
        # self.se = SEblock(bottleneck_planes)
 

    def forward(self, x):
        # identity = x
        identity = self.inconv(x)
        
        xs = torch.chunk(identity, 4, 1)
        # 用来将tensor分成很多个块，简而言之就是切分吧，可以在不同维度上切分。
        ys = []
        for s in range(4):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.x3conv(xs[s]))
            else:
                ys.append(self.x3conv(xs[s] + ys[s-1]))
        # out = torch.cat(ys, 1)
        # out = self.outconv(out)
        # out = self.se(out)
        # out += identity
        
        return ys
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attblock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attblock, self).__init__()
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

class PAM(nn.Module):

    def __init__(self, in_channels):
        # 有关键字参数，当传入字典形式的参数时，就要使用**kwargs
        
        super(PAM, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        # 让某些变量在学习的过程中不断的修改其值以达到最优化，初始值為 0
        self.softmax = nn.Softmax(dim=-1)
        # 是对某一维度的行进行softmax运算, https://blog.csdn.net/Will_Ye/article/details/104994504

    def forward(self, x):
        B, C, H, W = x.shape # "_"作为临时性的名称使用
        feat_b = self.conv_b(x).view(B, -1, H * W).permute(0, 2, 1)  # (N, H*W, C')
        feat_c = self.conv_c(x).view(B, -1, H * W) # (N, C', H*W)
        # view函数的作用为重构张量的维度
        # torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度
        energy = torch.bmm(feat_b, feat_c)  # (N, H*W, H*W)
        attention = self.softmax(energy) 
        # 计算两个tensor的矩阵乘法，第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样
        feat_d = self.conv_d(x).view(B, -1, H * W)  # (N, C, H*W)
        
        out = torch.bmm(feat_d, attention.permute(0, 2, 1))
        out = out.view(B, -1, H, W)
        out = self.alpha * out + x

        return out
    
class CAM(nn.Module):

    def __init__(self):
        super(CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        # β尺度係數初始化爲0，並逐漸地學習分配到更大的權重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        feat_a = x.view(B, C, -1)  # (N, C, H*W)
        feat_a_transpose = x.view(B, C, -1).permute(0, 2, 1)  # (N, H*W, C)
        attention = torch.bmm(feat_a, feat_a_transpose)  # (N, C, C)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        # torch.max(input, dim, keepdim)[0]: 只返回最大值
        # 1.input 是输入的tensor。
        # 2.dim 是索引的维度，dim=0寻找每一列的最大值，dim=1寻找每一行的最大值。
        # 3.keepdim 表示是否需要保持输出的维度与输入一样，keepdim=True表示输出和输入的维度一样，keepdim=False表示输出的维度被压缩了，也就是输出会比输入低一个维度。
        # a.expand_as(b) ：把tensor a扩展成和 b一样的形状
        attention = self.softmax(attention_new)
        value = x.view(B, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)
        out = self.beta * out + x

        return out
    
class DANet(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(DANet, self).__init__()
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

        self.pam = PAM(in_channels=F_int)
        self.cam = CAM()
    def forward(self, g, x):
        
        g1 = self.W_g(g)  # 下采样的gating signal 卷积
        x1 = self.W_x(x)  # 上采样的 l 卷积
        psi = self.relu(g1 + x1)
       
        cam = self.cam(psi)
        pam = self.pam(psi)
        da = self.psi(cam * pam) # channel 减为1，并Sigmoid,得到权重矩阵

        return x * da

###SANET
def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    # 因为要分组，先 reshape 成5个维度
    chnls_per_group = c // groups
    x = x.reshape(b, groups, chnls_per_group, h, w)
    # 把 groups 和 channel 维度替换
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # 恢复成输入的形状，实现 channel shuffle
    x = x.reshape(b, -1, h, w)
    return x

class SANet(nn.Module):
    def __init__(self, F_g, F_l, F_int, groups=4):
        super(SANet, self).__init__()
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

    #    self.pam = _PositionAttentionModule(F_int//(2*groups))
    #    self.cam = _ChannelAttentionModule()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, F_int // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, F_int // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, F_int // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, F_int // (2 * groups), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(F_int // (2 * groups), F_int // (2 * groups))

        self.groups = groups
        #self.channel_shuffle = channel_shuffle(F_int//(2*groups), 4)
    def forward(self, g, x):
        
        g1 = self.W_g(g)  # 下采样的gating signal 卷积
        x1 = self.W_x(x)  # 上采样的 l 卷积
        psi = self.relu(g1 + x1) #512
        
        b, c, h, w = psi.shape
        # 将各个组与 n 合并在一维
        psi = psi.reshape(b * self.groups, -1, h, w)
        # 每组特征拆成 2 组，方便 2 分支处理
        psi_0, psi_1 = psi.chunk(2, dim=1)
#         cam = self.cam(psi_0) #64
#         pam = self.pam(psi_1) #64

        # channel attention
        xn = self.avg_pool(psi_0)
        xn = self.cweight * xn + self.cbias
        xn = psi_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(psi_1)
        xs = self.sweight * xs + self.sbias
        xs = psi_1 * self.sigmoid(xs)
        
        da = torch.cat([xs, xn], dim=1) #128
        da = da.reshape(b, -1, h, w)
        da = channel_shuffle(da,4)
        #da = self.psi(da)
        
        return x*da
    
    
# Residual dense network
class RDB(nn.Module):
    def __init__(self, num_channels):
        super(RDB, self).__init__()
        self.conv_re = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True))
        
        self.lff = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = x + self.conv_re(x)
        x2 = x + x1 + self.conv_re(x1)
        x3 = x + x1 + x2 + self.conv_re(x2)
        
        out = self.lff(x3)
        return x + out
    
class RDN(nn.Module):
    def __init__(self, num_channels, nFeat):
        super(RDN, self).__init__()

        # F-1
        self.conv1 = nn.Conv2d(num_channels, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3 
        self.RDB1 = RDB(nFeat)
        self.RDB2 = RDB(nFeat)
        self.RDB3 = RDB(nFeat)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
       
        self.conv3 = nn.Conv2d(nFeat, num_channels, kernel_size=3, padding=1, bias=True)
    def forward(self, x):

        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)     
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_

        output = self.conv3(FDF)

        return output
    
    
class RDNRDN(nn.Module):
    def __init__(self, in_channels, num_channels):
        super(RDNRDN, self).__init__()
      
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True))
        
        self.conv_ir = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True))
        
        self.conv =nn.Sequential(
            nn.Conv2d(128, num_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_channels))
        
        self.conv_=nn.Sequential(
            nn.Conv2d(192, num_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_channels))
        
        self.rdb = RDB(num_channels)
        
            
    def forward(self, x):
        rgb = x[:,0:3,:,:]
        ir = x[:,3,:,:].unsqueeze(1)
       
        rgb1 = self.conv_rgb(rgb)
        ir1 = self.conv_ir(ir)
        cat1 = torch.cat((rgb1, ir1), 1)
        ca1 = self.conv(cat1)
       
        
        rgb2 = self.rdb(ca1)
        ir2 = self.rdb(ir1)
        cat2 = torch.cat((rgb2, ir2), 1)
        ca2 = self.conv(cat2)
        
        rgb3 = self.rdb(ca2)
        ir3 = self.rdb(ir2)
        cat3 = torch.cat((rgb3, ir3), 1)
        ca3= self.conv(cat3)
        
        rgb4 = self.rdb(ca3)
        ir4 = self.rdb(ir3)
        cat4 = torch.cat((rgb4, rgb3, rgb2), 1)
        ir_cat = torch.cat((ir4, ir3, ir2), 1)
        
        rgb5 = self.conv_(cat4)
        ir5 = self.conv_(ir_cat)
        
        rgb_fin = rgb5+rgb1
        ir_fin = ir5+ir1
        rgbir_rdn = torch.cat((rgb_fin, ir_fin), 1)
        out = self.conv(rgbir_rdn)
        
        return out
    
class RGBIR_RDN(nn.Module):
    def __init__(self, in_channels, num_channels=64, nFeat=64):
        super(RGBIR_RDN, self).__init__()
        self.rdnrdn = RDNRDN(in_channels, num_channels)
        self.rdn = RDN(num_channels, nFeat)
        
    def forward(self, x):
        x = self.rdnrdn(x)
        x = self.rdn(x)
        return x
    
class RGB2HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB2HSV, self).__init__()
        self.eps = eps
        
    def forward(self, x):
        x = x[:,:3,:,:]
        hue = torch.Tensor(x.shape[0], x.shape[2],x.shape[3]).to(x.device)
        
        hue[x[:,2]==x.max(1)[0]] = 4.0 + ((x[:,0]-x[:,1])/(x.max(1)[0]-x.min(1)[0]+self.eps)) [x[:,2]==x.max(1)[0]]
        hue[x[:,1]==x.max(1)[0]] = 2.0 + ((x[:,2]-x[:,0])/(x.max(1)[0]-x.min(1)[0]+self.eps)) [x[:,1]==x.max(1)[0]]
        hue[x[:,0]==x.max(1)[0]] = 0.0 + ((x[:,1]-x[:,2])/(x.max(1)[0]-x.min(1)[0]+self.eps)) [x[:,0]==x.max(1)[0]]%6
        
        hue[x.min(1)[0]==x.max(1)[0]] = 0.0
        hue = hue/6
        h = hue.unsqueeze(1)
        return h
    
######
def _make_layer(self, block, planes, blocks, stride=1, rate=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, rate, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, len(blocks)):
        layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

    return nn.Sequential(*layers)


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
       
    
    
    
###### ViT #####    

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.num_heads = head_num
        self.scale = (embedding_dim // head_num) ** (1 / 2)
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.qkv_convproj = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, groups = embedding_dim, bias=False),
            nn.BatchNorm2d(embedding_dim),
            Rearrange('b c h w -> b (h w) c'))
        
        self.proj_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, h, w):
        # print('x', x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # print('x', x.shape)
        
        q = self.qkv_convproj(x)
        k = self.qkv_convproj(x)
        v = self.qkv_convproj(x)
        # print("q", q.shape)

        query = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        key = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        value = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        att_score = torch.einsum("bhlk, bhtk -> bhlt", query, key) * self.scale
        attention = torch.softmax(att_score, dim=-1)
        x = torch.einsum("bhlt, bhtv -> bhlv", attention, value)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.out_attention(x)
        x = self.drop(x)
        return x
    
class MHA_Res2proj(nn.Module):
    def __init__(self, in_ch, head_num) -> None:
        super().__init__()
        self.num_hrads = head_num
        self.res2 = Res2Net(in_ch, in_ch)
        self.proj_q = nn.Linear(in_ch//4, in_ch, bias=False)
        self.proj_k = nn.Linear(in_ch//4, in_ch, bias=False)
        self.proj_v = nn.Linear(in_ch//4, in_ch, bias=False)

        self.out_attention = nn.Linear(in_ch*4, in_ch, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, h, w):
        # 定義需要重複使用的變量
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        res2 = self.res2(x)
        rearrange_list = [rearrange(res2[i], 'b c h w -> b (h w) c') for i in range(4)]
        q_list = [rearrange(self.proj_q(rearrange_list[i]), 'b t (h d) -> b h t d', h=self.num_hrads) for i in range(4)]
        k_list = [rearrange(self.proj_k(rearrange_list[i]), 'b t (h d) -> b h t d', h=self.num_hrads) for i in range(4)]
        v_list = [rearrange(self.proj_v(rearrange_list[i]), 'b t (h d) -> b h t d', h=self.num_hrads) for i in range(4)]

        att_list = [torch.einsum("bhlk, bhtk -> bhlt", q_list[i], k_list[i]) for i in range(4)]
        att_score_list = [torch.softmax(att_list[i], dim=-1) for i in range(4)]
        x_list = [torch.einsum("bhlt, bhtv -> bhlv", att_score_list[i], v_list[i]) for i in range(4)]

        # 合併所有的 x
        x = torch.cat([rearrange(x_list[i], 'b h t d -> b t (h d)') for i in range(4)], dim=2)
        x = self.out_attention(x)
        x = self.drop(x)
        return x # (b, (h*w), c)

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MHA_Res2proj(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, h, w):
        _x = self.multi_head_attention(x, h, w)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x, h, w):
        for layer_block in self.layer_blocks:
            x = layer_block(x, h, w)

        return x


class ViT(nn.Module):
    def __init__(self, in_c, out_ch, img_size, head_num=4, mlp_dim=1024, block_num=6, patch_size=1):
        super().__init__()

        self.img_dim = img_size
        self.inc = in_c
        # self.patch_dim = patch_dim

        self.num_tokens = (img_size // patch_size) ** 2
        self.token_dim = in_c * (patch_size ** 2)

#         #　linear
#         self.projection = nn.Linear(self.token_dim, self.token_dim)

        # convemb
        self.conv_proj = nn.Conv2d(in_c, out_ch, kernel_size=patch_size, stride=patch_size,padding=0)

        # self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, self.token_dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(self.token_dim, head_num, mlp_dim, block_num)
        
        # self.conv = nn.Conv2d(in_c, out_ch, kernel_size=1)

    def forward(self, x):
#         img_patches = rearrange(x, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)', patch_x=self.patch_dim, patch_y=self.patch_dim)

#         batch_size, tokens, _ = img_patches.shape
#         project = self.projection(img_patches)
#         token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
#                        batch_size=batch_size)

#         patches = torch.cat([token, project], dim=1)
#         patches += self.embedding[:tokens + 1, :]

#         x = self.dropout(patches)
#         x = self.transformer(x)
#         x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        x = self.conv_proj(x)
        B, C, H, W = x.shape  # (B, C, H=h/patch_size, W=w/patch_size)
        # print('x', x.shape)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = self.dropout(x)
        x = self.transformer(x, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x
    
    
###### cvt #######

import torch
from torch import nn, einsum
from einops import rearrange
from opt_einsum import contract


class SepConv2d(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=in_ch)
        
        self.bn = torch.nn.BatchNorm2d(in_ch)
        self.pointwise = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.1):

        super().__init__()
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        # self.avg_pool = nn.AdaptiveAvgPool2d(self.img_size//4)
        # self.max_pool = nn.AdaptiveMaxPool2d(self.img_size//4)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MSAttention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.1):
        super().__init__()

        self.img_size = img_size
        # inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.res2 = Res2Net(dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_feature = int(self.img_size//4)**2#//(self.img_size//4)

        self.out_attention = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # print('x',x.shape)
        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        res2 = self.res2(x)
        
        # chunk = [torch.chunk(res2[i], 2, dim=1) for i in range(4)]
        # pool = [torch.cat((self.avg_pool(chunk[i][0]), self.max_pool(chunk[i][1])), dim=1) for i in range(4)]
     
        q_list = [rearrange(res2[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        k_list = [rearrange(res2[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        v_list = [rearrange(res2[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        
        att_list = [contract('b h i d, b h j d -> b h i j', q_list[i], k_list[i])*self.scale for i in range(4)]
        # print('*',att_list[0].shape)
        att_score_list = [torch.softmax(att_list[i], dim=-1) for i in range(4)]
        # print('a',att_score_list[0].shape)
        x_list = [contract('b h i j, b h j d -> b h i d', att_score_list[i], v_list[i]) for i in range(4)]
        # print('av',x_list[0].shape)

        # 合併所有的 x
        x = torch.cat([rearrange(x_list[i], 'b h n d -> b n (h d)') for i in range(4)], dim=2).to(torch.float)
        x = self.out_attention(x)
        x = self.drop(x)
        return x # (b, (h*w), c)
    

    
class Res2Attention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.1):
        super().__init__()

        self.img_size = img_size
        # inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.res2 = Res2Net(dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_feature = int(self.img_size//4)**2#//(self.img_size//4))
        self.weight = nn.Parameter(torch.ones(1,1,1,self.num_feature)) 
        self.hue = RGB2HSV()
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim//4, dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim//4, dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim//4, dim, kernel_size, v_stride, pad)
        

        
        self.avg_pool = nn.AdaptiveAvgPool2d(self.img_size//4)
        self.max_pool = nn.AdaptiveMaxPool2d(self.img_size//4)
        self.out_attention = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # print('x',x.shape)
        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        res2 = self.res2(x)
        
        chunk = [torch.chunk(res2[i], 2, dim=1) for i in range(4)]
        pool = [torch.cat((self.avg_pool(chunk[i][0]), self.max_pool(chunk[i][1])), dim=1) for i in range(4)]
     
        q_list = [rearrange(res2[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        k_list = [rearrange(pool[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        v_list = [rearrange(pool[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        # print('q_list', q_list[0])
        # print('k_list', k_list[0])
        
        att_list = [contract('b h i d, b h j d -> b h i j', q_list[i], k_list[i])*self.scale for i in range(4)]
        # print('*',att_list[0].shape)
        att_score_list = [torch.softmax(att_list[i], dim=-1) for i in range(4)]
        # print('a',att_score_list[0].shape)
        x_list = [contract('b h i j, b h j d -> b h i d', att_score_list[i], v_list[i]) for i in range(4)]
        # print('av',x_list[0].shape)
   

        # 合併所有的 x
        x = torch.cat([rearrange(x_list[i], 'b h n d -> b n (h d)') for i in range(4)], dim=2).to(torch.float)
        x = self.out_attention(x)
        x = self.drop(x)
        return x # (b, (h*w), c)


##### new ###

class Att_rgbih(nn.Module):
    def __init__(self, F_int):
        super(Att_rgbih, self).__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, h):
        psi = self.relu(x + h)
        psi = self.psi(psi) 

        return x * psi
    
class adjusted_cosine_similarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x, y):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        y = y - torch.mean(y, dim=-1, keepdim=True)
        cos_sim = self.cos_sim(x,y)
        var_x = torch.var(x, dim=-1, unbiased=True)
        var_y = torch.var(y, dim=-1, unbiased=True)
        denom = torch.sqrt(var_x * var_y)
        return (cos_sim * (denom / torch.clamp(denom, min=1e-12))+1)/2
    
class new_sim(nn.Module):
    def __init__(self, dim, heads, eps=1e-12):
        super().__init__()
        self.eps= eps
        self.num_feature = int(dim/heads)//4
        self.weight = nn.Parameter(torch.ones(1,1,1,self.num_feature))  # 轉換參數為GPU張量
        self.n = dim**2
        # self.weight = nn.Parameter(torch.ones(1,1,self.n,1)) 
        
    def forward(self, x, y):
        # d = torch.cdist(x,y)
        
        x_ = x.cpu().detach().numpy()
        y_ = y.cpu().detach().numpy()
        x_ = cp.asarray(x_)
        y_ = cp.asarray(y_)
        diff = x_-y_
        dist = cp.sqrt(self.n * diff**2)
        j = 1-cp.log10(dist+1)
        j = torch.tensor(j).to(x.device)
        return j
    
# def scale(heads):
#     n = torch.ones(heads, 1, 1)*10
#     n = n.cpu().detach().numpy()
#     n = cp.asarray(n)
#     logit_scale = nn.Parameter(torch.tensor(cp.log(n)).to(x.device), requires_grad=True) 
#     logit_scale = torch.clamp(logit_scale.to(x.device), max=torch.log(torch.tensor(100)).to(x.device)).exp()
#     return logit_scale
        