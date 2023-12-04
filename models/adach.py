import torch
import torch.nn as nn
import numpy as np

from models import flooding_model
# from models.gt import gt_batch

# import ml4floods.data.worldfloods.configs  
# from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS, CHANNELS_CONFIGURATIONS, BANDS_S2, COLORS_WORLDFLOODS_INVLANDWATER, COLORS_WORLDFLOODS_INVCLEARCLOUD

class weight():     
    def rgb2hsv_torch(self, rgb: torch.Tensor) -> torch.Tensor:
            # rgb = rgb.unsqueeze(0).permute(0,3,1,2)
            cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
            cmin = torch.min(rgb, dim=1, keepdim=True)[0]
            # print("cmin", cmin.device)
            delta = cmax - cmin
            hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
            # print("hsv_h", hsv_h.device)
            cmax_idx[delta == 0] = 3
            hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
            hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
            hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
            hsv_h[cmax_idx == 3] = 0.
            hsv_h /= 6.
            return hsv_h

    def bin_IR(self, ir):  # 將IR二值化，[0~1] --> (th>0.2)為黑/(th<0.2)為白
        ir = torch.tensor(ir).cuda()
        th = 0.2
        ir = ir < th
        ir = ir.to(torch.float)

        return ir   # array

    def bin_H(self, h, threshold = 0.2):   # 將H二值化，[0~1] --> (th>0.2)為0(th<0.2)為1
        H = h < threshold
        H = H.to(torch.float)
        return H   # tensorfloat

    def bin_GT(self, gt):   # water[0, 0, 139] --> [1,1,1]，other --> [0,0,0]

        th1 = 1
        th2 = 2
        gt1 = gt > th1  # (0,1) --> 0
        gt2 = gt > th2  # (0,1,2) --> 0, (3) --> 1
        gt1 = gt1.to(torch.float)
        gt2 = gt2.to(torch.float)
        gt = gt1-gt2
        return gt   # array

    def IoU_weight(self, img, gt_b, ir_b, h_b, ir, h):

        if torch.all(gt_b==0):
            w_ir = 0.01
            w_h =0.01
        else:
            intersection_ir = torch.sum(torch.logical_and(gt_b, ir_b))
            union_ir = torch.sum(torch.logical_or(gt_b, ir_b))
            # h_s= h.squeeze().squeeze()
            intersection_h = torch.sum(torch.logical_and(gt_b, h_b))
            union_h = torch.sum(torch.logical_or(gt_b, h_b))
            
            #Intersection over Union
            iou_ir = intersection_ir / union_ir
            iou_h = intersection_h / union_h
            
            if iou_ir >= 0.5:
                init_range = 0.1
                w_ir = nn.init.xavier_uniform_(torch.rand(1,1), init_range)
            else:
                init_range = 0.01
                w_ir = nn.init.xavier_uniform_(torch.rand(1,1), init_range)

            if iou_h >= 0.5:
                init_range = 0.1
                w_h = nn.init.xavier_uniform_(torch.rand(1,1), init_range)
            else:
                init_range = 0.01
                w_h = nn.init.xavier_uniform_(torch.rand(1,1), init_range)
                
#             if (iou_h == 0 and iou_ir !=0):
#                 w_h = 0.01
#                 w_ir = iou_ir / (iou_h + iou_ir)
#             elif (iou_h != 0 and iou_ir ==0):
#                 w_ir =0.01
#                 w_h = iou_h / (iou_h + iou_ir)
#             elif (iou_h == 0 and iou_ir ==0):
#                 w_ir = 0.01
#                 w_h =0.01
#             else:
#                 w_h = iou_h / (iou_h + iou_ir)
#                 w_ir = iou_ir / (iou_h + iou_ir)
            
        w_ir = torch.tensor(w_ir).cuda()
        w_h = torch.tensor(w_h).cuda()
        img = torch.tensor(img).cuda()
        ir = torch.tensor(ir).cuda()
        h = torch.tensor(h).cuda()


        x = torch.cat((img, ir*w_ir, h*w_h), dim = 1)
        return x

class AdaCh(nn.Module):  # Hunet
    def __init__(self):
        super(AdaCh, self).__init__()
        self.gt = None

    def forward(self, img, gt):

        self.gt = gt
        w = weight()
        rgb = img[:,:3,:,:]
        ir = img[:,3:,:,:]
        # print('ir_tor', ir, torch.max(ir))
        ir = flooding_model.unnorm_batch(img, channel_configuration = "bgri")
        ir = ir[:,3:,:,:]
        # print('ir_np', ir, np.max(ir))
        h = w.rgb2hsv_torch(rgb)

        GT_b = w.bin_GT(gt).cuda()
        IR_b = w.bin_IR(ir).cuda()
        H_b = w.bin_H(h).cuda()
        Ada_Ch = w.IoU_weight(rgb, GT_b, IR_b, H_b, ir, h)
        return Ada_Ch