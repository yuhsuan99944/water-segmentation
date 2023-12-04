import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np

from models.Compare_architecture import Mlp, window_partition, window_reverse, PatchEmbed, PatchExpand, PatchMerging, BasicLayer_up, BasicLayer, FinalPatchExpand_X4
from models.Compare_architecture import ASPP, MobileNetV2
from models.Compare_architecture import ConvBNReLU,  EAmodule, DecoderStem, Stem
from models.Compare_architecture import BasicBlock, BasicTransBlock, up_block, up_block_trans, down_block, down_block_trans


class DeepLab_V3_plus(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":

            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
      
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))


        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


######## Swin Unet #########
class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, in_chans, num_classes, img_size=256, patch_size=4, 
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            # print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)


       #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
  
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x)

        return x

# class SwinUnet(nn.Module):
#     def __init__(self,num_classes, img_size=256 , zero_head=False, vis=False):
#         super(SwinUnet, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.config = config

#         self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
#                                 patch_size=config.MODEL.SWIN.PATCH_SIZE,
#                                 in_chans=config.MODEL.SWIN.IN_CHANS,
#                                 num_classes=self.num_classes,
#                                 embed_dim=config.MODEL.SWIN.EMBED_DIM,
#                                 depths=config.MODEL.SWIN.DEPTHS,
#                                 num_heads=config.MODEL.SWIN.NUM_HEADS,
#                                 window_size=config.MODEL.SWIN.WINDOW_SIZE,
#                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                 qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#                                 qk_scale=config.MODEL.SWIN.QK_SCALE,
#                                 drop_rate=config.MODEL.DROP_RATE,
#                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                 ape=config.MODEL.SWIN.APE,
#                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)

#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         logits = self.swin_unet(x)
#         return logits


########## MTUnet ##########

class encoder_block(nn.Module):
    def __init__(self, dim):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim),
            EAmodule(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1,
                                       2)  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)  # (14, 14, 256)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, dim, flag):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2),
                EAmodule(dim // 2)
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                EAmodule(dim),
                EAmodule(dim)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MTUNet, self).__init__()
        self.stem = Stem(in_ch)
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(1024),
                                        EAmodule(1024))
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len([256,512])):
            dim = [256,512][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len([1024, 512]) - 1):
            dim = [1024, 512][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block([1024, 512][-1], True))
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x


######### UTnet ########

class UTNet(nn.Module):
    
    def __init__(self, in_chan, num_classes, base_chan=32, reduce_size=8, block_list='234', num_blocks=[1, 2, 4], projection='interp', num_heads=[2,4,8], attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan//num_heads[-5], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2*base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4], dim_head=base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        else:
            self.inc.append(BasicBlock(base_chan, base_chan))
            self.up4 = up_block(2*base_chan, base_chan, scale=(2,2), num_block=2)
        self.inc = nn.Sequential(*self.inc)


        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2*base_chan, num_block=num_blocks[-4], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], dim_head=2*base_chan//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up3 = up_block_trans(4*base_chan, 2*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-3], dim_head=2*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, 2*base_chan, (2,2), num_block=2)
            self.up3 = up_block(4*base_chan, 2*base_chan, scale=(2,2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(2*base_chan, 4*base_chan, num_block=num_blocks[-3], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3], dim_head=4*base_chan//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up2 = up_block_trans(8*base_chan, 4*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-2], dim_head=4*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.down2 = down_block(2*base_chan, 4*base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8*base_chan, 4*base_chan, scale=(2,2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(4*base_chan, 8*base_chan, num_block=num_blocks[-2], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2], dim_head=8*base_chan//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            self.up1 = up_block_trans(16*base_chan, 8*base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-1], dim_head=8*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.down3 = down_block(4*base_chan, 8*base_chan, (2,2), num_block=2)
            self.up1 = up_block(16*base_chan, 8*base_chan, scale=(2,2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(8*base_chan, 16*base_chan, num_block=num_blocks[-1], bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1], dim_head=16*base_chan//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.down4 = down_block(8*base_chan, 16*base_chan, (2,2), num_block=2)


        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        if aux_loss:
            self.out1 = nn.Conv2d(8*base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4*base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2*base_chan, num_classes, kernel_size=1, bias=True)
            


    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        if self.aux_loss:
            out = self.up1(x5, x4)
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:
            out = self.up1(x5, x4)
            out = self.up2(out, x3)
            out = self.up3(out, x2)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out




