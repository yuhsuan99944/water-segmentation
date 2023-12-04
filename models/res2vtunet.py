import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
from einops import rearrange
from opt_einsum import contract

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
        

class Res2Net(nn.Module):
    expansion = 4  #輸出channel=輸入channel*expansion

    def __init__(self, in_ch, out_ch, downsample=None, stride=1, scales=4, se=False,  norm_layer=None):
        super(Res2Net, self).__init__()
        if out_ch % scales != 0:   # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        
        self.inconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch))
          
        self.x3conv = nn.Sequential(
            nn.Conv2d(out_ch // scales, out_ch // scales,  kernel_size=3, stride=stride, padding=1, groups=out_ch // scales, bias=False),
            nn.Conv2d(out_ch // scales, out_ch // scales, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch // scales))

    def forward(self, x):
        # identity = x
        identity = self.inconv(x)
        
        xs = torch.chunk(identity, 4, 1)
        ys = []
        for s in range(4):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.x3conv(xs[s]))
            else:
                ys.append(self.x3conv(xs[s] + ys[s-1]))
        
        return ys
        

class MLP(nn.Module):
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
        
 
        self.linear =nn.Linear(1, (dim//heads)//4, bias=False)
        self.ln_v =nn.Linear((dim//4)//heads, (self.img_size//4)**2, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(self.img_size//4)
        self.max_pool = nn.AdaptiveMaxPool2d(self.img_size//4)
        self.out_attention = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        res2 = self.res2(x)
        
        chunk = [torch.chunk(res2[i], 2, dim=1) for i in range(4)]
        pool = [torch.cat((self.avg_pool(chunk[i][0]), self.max_pool(chunk[i][1])), dim=1) for i in range(4)]
     
        q_list = [rearrange(res2[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        k_list = [rearrange(pool[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        v_list = [rearrange(pool[i], 'b (h d) l w -> b h (l w) d', h=self.heads) for i in range(4)]
        
        att_list = [contract('b h i d, b h j d -> b h i j', q_list[i], k_list[i])*self.scale for i in range(4)]
        att_score_list = [torch.softmax(att_list[i], dim=-1) for i in range(4)]
        x_list = [contract('b h i j, b h j d -> b h i d', att_score_list[i], v_list[i]) for i in range(4)]   

        # 合併所有的 x
        x = torch.cat([rearrange(x_list[i], 'b h n d -> b n (h d)') for i in range(4)], dim=2).to(torch.float)
        x = self.out_attention(x)
        x = self.drop(x)
        return x # (b, (h*w), c)                
        

class TransformerBlock(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.sz = img_size
        self.mha = Res2Attention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP(dim, mlp_dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        mha = self.mha(x)
        ln1 = self.ln1(mha)
        mha = self.dropout(ln1)
        x = mha + x + h
   
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
        
        x2, h2 = self.cvt1(x1*self.w1_rgbi, h*self.w1_h)  # torch.Size([1, 64, 64, 64])
        x3, h3 = self.cvt2(x2*self.w2_rgbi, h2*self.w2_h)
        
        bt, _ = self.cvt3(x3*self.w3_rgbi, h3*self.w3_h)
  
        x = nn.functional.interpolate(bt, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        d3 = self.dconv_up3(x)
        
        x = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        d2 = self.dconv_up2(x)
        
        x = nn.functional.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.dconv_up1(x)      
        
        x = self.dropout(x)
        out = self.conv_last(x)
        return out
    