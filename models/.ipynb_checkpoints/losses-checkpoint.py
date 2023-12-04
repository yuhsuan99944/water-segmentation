from typing import Optional

import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np
from math import exp
import cv2

from models import flooding_model
import warnings
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as TF

from models.boundary import SurfaceLoss, class2one_hot, one_hot2dist



def calc_loss_mask_invalid_original_unet(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = cross_entropy_loss_mask_invalid(logits, target, weight=weight)
    return bce

def dice_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
    """
    Dice loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, C, H, W) tensor with logits (no softmax)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        smooth: Value to avoid div by zero

    Returns:
        averaged loss (float)

    """
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    pred = torch.softmax(logits, dim=1)
    valid = (target != 0) # (B, H, W) tensor
    target_without_invalids = (target - 1) * valid  # Set invalids to land

    target_one_hot_without_invalid = torch.nn.functional.one_hot(target_without_invalids,
                                                                 num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    axis_red = (2, 3) # H, W reduction

    pred_valid = pred * valid.unsqueeze(1).float()  # # Set invalids to 0 (all values in prob tensor are 0

    intersection = (pred_valid * target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor

    union = pred_valid.sum(dim=axis_red) + target_one_hot_without_invalid.sum(dim=axis_red)  # (B, C) tensor

    dice_score = ((2. * intersection + smooth) /
                 (union + smooth))

    loss = (1 - dice_score)  # (B, C) tensor

    return torch.mean(loss)

def cross_entropy_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
        averaged loss
    """
#     print('trongan93-test losses: Cross entropy loss function for mask')
#     print(logits.dim())
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    # BCE Loss (ignoring invalid values)
    bce = F.cross_entropy(logits, target_without_invalids,
                          weight=weight, reduction='none')  # (B, 1, H, W)

    bce *= valid  # mask out invalid pixels > 屏蔽無效像素

    return torch.sum(bce / (torch.sum(valid) + 1e-6))



def calc_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = cross_entropy_loss_mask_invalid(logits, target, weight=weight)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    dice = dice_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return bce * bce_weight + dice * (1 - bce_weight)


def binary_cross_entropy_loss_mask_invalid(logits: torch.Tensor, target: torch.Tensor,
                                           pos_weight:Optional=None) -> float:
    """

    Args:
        logits: (B, H, W) tensor with logits (no signmoid!)
        target: (B, H, W) Tensor with values encoded as {0: invalid, 1: neg_xxx, 2:pos_xxx}
        pos_weight: weight of positive class

    Returns:

    """
    assert logits.dim() == 3, f"Unexpected shape of logits. Logits: {logits.shape} target: {target.shape}"
    assert target.dim() == 3, f"Unexpected shape of target. Logits: {logits.shape} target: {target.shape}"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    pixelwise_bce = F.binary_cross_entropy_with_logits(logits, target_without_invalids.float(), reduction='none',
                                                       pos_weight=pos_weight)

    pixelwise_bce *= valid  # mask out invalid pixels

    return torch.sum(pixelwise_bce / (torch.sum(valid) + 1e-6))

def calc_loss_multioutput_logistic_mask_invalid(logits: torch.Tensor, target: torch.Tensor,
                                                pos_weight_problem:Optional[List[float]]=None,
                                                weight_problem:Optional[List[float]]=None) -> float:

    assert logits.dim() == 4, "Unexpected shape of logits"
    assert target.dim() == 4, "Unexpected shape of target"

    if weight_problem is None:
        weight_problem = [ 1/logits.shape[1] for _ in range(logits.shape[1])]

    total_loss = 0
    for i in range(logits.shape[1]):
        pos_weight = torch.tensor(pos_weight_problem[i], device=logits.device) if pos_weight_problem is not None else None
        curr_loss = binary_cross_entropy_loss_mask_invalid(logits[:, i], target[:, i], pos_weight=pos_weight)
        total_loss += curr_loss*weight_problem[i]

    return total_loss


# Optimize by Trong-An Bui (trongan93@gmail.com) 
# - Change the Cross Entropy Loss to Focal Loss function

def focal_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None, gamma=2, alpha=.25) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
        averaged loss
    """
#     print('trongan93-test losses: Cross entropy loss function for mask')
#     print(logits.dim())
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    ce_loss = F.cross_entropy(logits, target_without_invalids,reduction='none',weight=weight)
    ce_loss *= valid # mask out invalid pixels
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha*(1 - pt) ** gamma * ce_loss)
    
    return torch.sum(focal_loss / (torch.sum(valid) + 1e-6))

def calc_loss_mask_invalid_2(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = focal_loss_mask_invalid(logits, target, weight=weight)
    #ce改focal

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    dice = dice_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return bce * bce_weight + dice * (1 - bce_weight)

# Optimize 2 by Trong-An Bui (trongan93@gmail.com) 
# - Change the Cross Entropy Loss to Focal Loss function
# - Different loss functions for RGB and Nir

def iou_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
    """
    IoU loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        smooth: Value to avoid div by zero
    Returns:
        averaged loss (float)
    """
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    pred = torch.softmax(logits, dim=1)
    valid = (target != 0) # (B, H, W) tensor
    target_without_invalids = (target - 1) * valid  # Set invalids to land

    target_one_hot_without_invalid = torch.nn.functional.one_hot(target_without_invalids,
                                                                 num_classes=pred.shape[1]).permute(0, 3, 1, 2)  #將tensor的维度换位
    axis_red = (2, 3) # H, W reduction

    pred_valid = pred * valid.unsqueeze(1).float()  # # Set invalids to 0 (all values in prob tensor are 0
    # print('p',pred_valid.shape)
    # print('t', target_one_hot_without_invalid)
    intersection = (pred_valid * target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor
    total = (pred_valid + target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor

    union = total - intersection
    iou_score = (intersection + smooth)/(union + smooth)
    loss = (1 - iou_score)  # (B, C) tensor
    return torch.mean(loss)

def calc_loss_mask_invalid_3(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """
    # print(f"Shape of logits: {logits.shape}")
    # logits_rgb = logits[:,0:2,:,:]
    # print(f"Shape of logits_rgb: {logits_rgb.shape}")
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=5, alpha=0.01)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    iou_loss = iou_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return fc * bce_weight + iou_loss * (1 - bce_weight)

###
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    # 所有元素的和等於1
    return gauss/gauss.sum()

def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 使用高斯函數生成一維張量
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 該一維張量與其轉置交叉相乘得到二維張量(這保持了高斯特性) # 增加兩個額外的維度，將其轉換爲四維
    window =  Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # .expand將引數中的列表合併到原列表的末尾
    # 返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor
    # Variable就是变量的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
    return window

def ssim(logits: torch.Tensor, target:torch.Tensor, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    
    logits = logits.type(torch.float).cpu()
    target = target.type(torch.float).cpu()
    
    if val_range is None:
        if torch.max(logits) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(logits) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    _, channel, height, width = logits.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)  # .to(logits.device)
    
    # target = torch.Tensor(target)
#     int size = 0
#     Tensor target = torch::tensor(size, dtype(int))
    target = torch.Tensor(target).unsqueeze(0)
    target = target.permute(1,0,2,3)
    mu1 = F.conv2d(logits, window, padding=padd, groups=channel)
    mu2 = F.conv2d(target, window, padding=padd, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(logits * logits, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv2d(logits * target, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    
    return ret.cuda() 

def calc_loss_mask_invalid_4(logits: torch.Tensor, target:torch.Tensor, bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    # ,bce_weight:float=0.33
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
    """
 
    target=TF.resize(target, size=logits.shape[2:])
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=5, alpha=0.01)
    iou_loss = iou_loss_mask_invalid(logits, target) # (B, C)

    compound_loss = fc * bce_weight + iou_loss * (1 - bce_weight)

    return  fc, iou_loss, compound_loss


def msssim(logits: torch.Tensor, target:torch.Tensor, window_size=11, size_average=True, val_range=None, normalize=False):
    device = logits.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    
    for _ in range(levels):
        sim, cs = ssim(logits, target, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        logits = F.avg_pool2d(logits, (2, 2))
        target = F.avg_pool2d(target, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
        
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


######## Boundary ####
def Canny (img):
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    kernel_size=3
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size), 0)
    low_threshold=0
    high_threshold=5
    edges = cv2.Canny(blur_gray.astype(np.uint8), low_threshold, high_threshold)
    return edges

def BCE_Canny_loss_mask_invalid(logits: torch.Tensor, target: torch.Tensor,
                                    bce_weight:float=0.5, pos_weight:Optional=None) -> float:
    """
    Args:
        logits: (B, H, W) tensor with logits (no signmoid!)
        target: (B, H, W) Tensor with values encoded as {0: invalid, 1: neg_xxx, 2:pos_xxx}
        pos_weight: weight of positive class
    Returns:
    """
    w_C = nn.Parameter(torch.tensor(bce_weight))
                                               
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).long().cpu()
    valid = (target != 0)
    target_without_invalids = (target-1) * valid

    imgs = []
    gts = []
    for img in prediction:
        image =  flooding_model.rgb_for_Canny(img)
        logits_c = Canny(image)
        imgs.append(logits_c//255)

    for gt in target_without_invalids:
        mask = flooding_model.rgb_for_Canny(gt)
        target_c = Canny(mask)
        gts.append(target_c//255)

    imgs = np.array(imgs)
    logits = torch.tensor(imgs).to(logits.device)
    gts = np.array(gts)
    target = torch.tensor(gts).to(logits.device)
 
    assert logits.dim() == 3, f"Unexpected shape of logits. Logits: {logits.shape} target: {target.shape}"
    assert target.dim() == 3, f"Unexpected shape of target. Logits: {logits.shape} target: {target.shape}"

    weight = torch.ones_like(logits, dtype=torch.float32)
    pos_class_weight = len(torch.where(target == 0)[0]) / torch.numel(target) # (36.60054169/(1.93445299+2.19400729))
    weight[target == 1] = pos_class_weight
    weight[target == 0] = 1 - pos_class_weight
    
    pixelwise_bce = F.binary_cross_entropy(logits.float(), target.float(), weight = weight, reduction='none')#*0.01    
    pixelwise_bce = pixelwise_bce/torch.max(pixelwise_bce+ 1e-8)
    valid = (target != 0)
    pixelwise_bce *= valid  # mask out invalid pixels
    loss = torch.sum(pixelwise_bce / (torch.sum(valid) + 1e-6))
    return w_C*loss.requires_grad_()
                                               
def Fc_Canny_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None, gamma=2, alpha=.25) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
        averaged loss
    """
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).long().cpu()
    valid = (target != 0)
    target_without_invalids = (target-1) * valid

    imgs = []
    gts = []
    for img in prediction:
        image =  flooding_model.rgb_for_Canny(img)
        logits_c = Canny(image)
        imgs.append(logits_c//255)

    for gt in target_without_invalids:
        mask = flooding_model.rgb_for_Canny(gt)
        target_c = Canny(mask)
        gts.append(target_c//255)

    imgs = np.array(imgs)
    logits = torch.tensor(imgs).to(logits.device)
    gts = np.array(gts)
    target = torch.tensor(gts).to(logits.device)

    # assert logits.dim() == 4, f"Expected 4D tensor logits"
    # assert target.dim() == 3, f"Expected 3D tensor target"

    weight = torch.ones_like(logits, dtype=torch.float32)
    pos_class_weight = len(torch.where(target == 0)[0]) / torch.numel(target) # (36.60054169/(1.93445299+2.19400729))
    weight[target == 1] = pos_class_weight
    weight[target == 0] = 1 - pos_class_weight
    

    ce_loss = F.binary_cross_entropy(logits.float(), target.float(), reduction='none', weight=weight)
    ce_loss = ce_loss/torch.max(ce_loss+ 1e-8)
    valid = (target != 0)
    ce_loss *= valid # mask out invalid pixels
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha*(1 - pt) ** gamma * ce_loss)

    return torch.sum(focal_loss / (torch.sum(valid) + 1e-6))


def Boundary_iou_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, smooth=1) -> float:
    """
    IoU loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        smooth: Value to avoid div by zero
    Returns:
        averaged loss (float)
    """
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).long()
    
    valid = (target != 0)
    target_without_invalids = (target-1) * valid
    pred_without_invalids = prediction * valid
    # target_zeros =torch.zeros_like(target_without_invalids)
    # target_zeros[target_without_invalids == 1] =1
    
    imgs = []
    gts = []
    kernel = np.ones((3,3), np.uint8)
    imgs = []
    gts = []
    for img in pred_without_invalids:
        image =  flooding_model.rgb_for_Canny(img)
        logits_c = Canny(image)
        logits_c = cv2.dilate(logits_c, kernel, iterations = 1)
        imgs.append(logits_c//255)

    for gt in target_without_invalids:
        mask = flooding_model.rgb_for_Canny(gt)
        target_c = Canny(mask)
        target_c = cv2.dilate(target_c, kernel, iterations = 1)
        gts.append(target_c//255)
        # gt = gt.cpu().numpy().astype(float)
        # erosion = cv2.erode(gt, kernel, iterations = 1)
        # gts.append(erosion)

    imgs = np.array(imgs)
    boundary_img = torch.tensor(imgs, dtype = torch.float32, device=logits.device).unsqueeze(1)
    # print('boundary_img',boundary_img.dtype)
    gts = np.array(gts)
    boundary_gt = torch.tensor(gts, dtype = torch.float32, device=logits.device).unsqueeze(1)
   
    
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    # valid = (target != 0) # (B, H, W) tensor
    # target_without_invalids = (target - 1) * valid  # Set invalids to land

    target_one_hot_without_invalid = torch.nn.functional.one_hot(target_without_invalids.long(),
                                                                 num_classes=probs.shape[1]).permute(0, 3, 1, 2)  #將tensor的维度换位
    # target_one_hot_without_invalid = target_one_hot_without_invalid[:, 1, :,:]
    # print('target_one_hot0',len(torch.where(target_one_hot_without_invalid[:,0,:,:] == 1)[0]))
    # print('target_one_hot1',len(torch.where(target_one_hot_without_invalid[:,1,:,:] == 1)[0]))
    # print('target_one_hot2',len(torch.where(target_one_hot_without_invalid[:,2,:,:] == 1)[0]))
    pred_valid = probs * valid.unsqueeze(1).float()  # # Set invalids to 0 (all values in prob tensor are 0

    axis_red = (2, 3)  # H, W reduction
    # # gt intersection
    intersection_gt = (boundary_gt * target_one_hot_without_invalid) # (B, C) tensor

    # # pred intersection
    intersection_pred = (boundary_img * pred_valid)#.sum(dim=axis_red) # (B, C) tensor

    # Fn IoU
    intersection = (intersection_pred * intersection_gt).sum(dim=axis_red) # (B, C) tensor
    total = (intersection_pred + intersection_gt).sum(dim=axis_red) # (B, C) tensor

    union = total - intersection
    iou_score = (intersection + smooth)/(union + smooth)
    loss = (1 - iou_score)  # (B, C) tensor
    return torch.mean(loss)

def calc_loss_fc_biou(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """
    w_fc = nn.Parameter(torch.tensor(0.5))
    w_biou = nn.Parameter(torch.tensor(0.5))
    w_dice = nn.Parameter(torch.tensor(0.5))
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=5, alpha=0.01).requires_grad_()
    biou = Boundary_iou_loss_mask_invalid(logits, target) # (B, C)
    iou = iou_loss_mask_invalid(logits, target) 
    dice = dice_loss_mask_invalid(logits, target)

    loss = biou * w_biou + dice * w_dice 
   
    # Weighted sum
    return loss, w_biou, w_dice


# import matplotlib.pyplot as plt
# def Hausdorff_Canny_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor,bce_weight:float=1., 
#                                           weight:Optional[torch.Tensor]=None, eps = 1e-6, alpha=4) -> float:
#     """
#     F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
#     Args:
#         logits: (B, C, H, W) tensor with logits (no softmax!)
#         target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
#         weight: (C, ) tensor. weight per class value to cross_entropy function
#     Returns:
#         averaged loss
#     """
#     w_C = nn.Parameter(torch.tensor(bce_weight))
#     B, C, H, W = logits.shape
#     probs = torch.softmax(logits, dim=1)
#     prediction = torch.argmax(probs, dim=1)
#     print('pred',torch.max(prediction))
#     valid = (target != 0)
#     target_without_invalids = (target-1) * valid
#     pred_without_invalids = prediction * valid
#     print('t',torch.max(target_without_invalids))
#     print('p',torch.max(pred_without_invalids))
                                              
#     kernel = np.ones((3,3), np.uint8)
#     imgs = []
#     gts = []
#     for img in pred_without_invalids:
#         image =  flooding_model.rgb_for_Canny(img)
#         logits_c = Canny(image)
#         logits_c = cv2.dilate(logits_c, kernel, iterations = 1)
#         # plt.title('l')
#         # plt.imshow(logits_c)
#         imgs.append(logits_c//255)

#     for gt in target_without_invalids:
#         mask = flooding_model.rgb_for_Canny(gt)
#         target_c = Canny(mask)
#         target_c = cv2.dilate(target_c, kernel, iterations = 1)
#         # plt.title('gt')
#         # plt.imshow(target_c)
#         gts.append(target_c//255)

#     imgs = np.array(imgs)
#     logits = torch.tensor(imgs).to(logits.device).float()
#     gts = np.array(gts)
#     target = torch.tensor(gts).to(logits.device).float()
    
#     assert logits.dim() == 3, f"Expected 3D tensor logits"
#     assert target.dim() == 3, f"Expected 3D tensor target"
    
#     gt_2class = class2one_hot(target, 2)  # (b, num_class, h, w)
#     gt_2class = gt_2class[0].cpu().numpy()  
#     gt_dist = one_hot2dist(gt_2class)  # bcwh

#     gt_dist = torch.tensor(gt_dist).unsqueeze(0)
#     img_2class = class2one_hot(logits, 2)

#     Loss = SurfaceLoss()
#     res = Loss(img_2class, gt_dist, None)
#     # print('res',res)
    
#     return w_C*res.requires_grad_()


def Hausdorff_Canny_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor,bce_weight:float=0.2, 
                                          weight:Optional[torch.Tensor]=None, eps = 1e-6, alpha=4) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
        averaged loss
    """
    w_C = nn.Parameter(torch.tensor(bce_weight))
    B, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1)
                                          
    logits_w = torch.zeros_like(prediction, dtype=torch.bool)
    logits_w[prediction == 1]=True
    # print("l",logits_w.shape)
    valid = (target != 0)
    target = (target-1) * valid
    # pred_without_invalids = prediction * valid
    # print('t',torch.max(target_without_invalids))
    target_w = torch.zeros_like(target, dtype=torch.bool)
    target_w[target == 1]=True
    # print("t",target_w.shape)
    
    assert logits_w.dim() == 3, f"Expected 3D tensor logits"
    assert target_w.dim() == 3, f"Expected 3D tensor target"
    
    gt_2class = class2one_hot(target_w, 2)  # (b, num_class, h, w)
    gt_2class = gt_2class[0].cpu().numpy()  
    gt_dist = one_hot2dist(gt_2class)  # bcwh

    gt_dist = torch.tensor(gt_dist).unsqueeze(0)
    img_2class = class2one_hot(logits_w, 2)

    Loss = SurfaceLoss()
    res = Loss(img_2class, gt_dist, None)
    # print('res',res)
    
    return w_C*res.requires_grad_()                                            

# from scipy.ndimage import distance_transform_edt as eucl_distance
# def one_hot2hd_dist(seg: torch.Tensor, dtype=torch.float32) -> float:
#     res = torch.zeros_like(seg)

#     for cla in range(len(seg)):
#         water = torch.where(seg[cla] == 2, 1, 0)
#         if water.any():
#             water = water.cpu().numpy()
#             dist = eucl_distance(water)
#             dist = torch.tensor(dist,dtype = dtype, device=seg.device)#.unsqueeze(0)
#             res[cla] = dist
#             # print(res.dtype)

#     return res

# def HausdorffLoss(logits: torch.Tensor, target: torch.Tensor) -> float:
#     # assert simplex(probs)
#     # assert simplex(target)
#     # assert probs.shape == target.shape
#     target = target.unsqueeze(1)
#     valid = (target != 0)
#     # print(target.shape, target)

#     probs = torch.softmax(logits, dim=1)
#     B, C, H, W = probs.shape  # type: ignore
    
#     # pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
#     # tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#     # assert pc.shape == tc.shape == (B, len(self.idc), *xyz)
    
#     tdm = torch.stack([one_hot2hd_dist(target[b]).float() for b in range(B)], dim=0)
#     # print(tdm.dtype)
#     pred= torch.argmax(probs, dim=1)
#     pdm = torch.stack([one_hot2hd_dist(pred[b]).float() for b in range(B)], dim=0)
    
#     dtm = (tdm**2 + pdm**2)**0.5
#     dtm = dtm*valid.float()
    
#     loss = dtm.mean()
#     # print(loss)
#     return loss

# def Hausdorff_Canny_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, p=2) -> float:
#     """
#     F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)
#     Args:
#         logits: (B, C, H, W) tensor with logits (no softmax!)
#         target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
#         weight: (C, ) tensor. weight per class value to cross_entropy function
#     Returns:
#         averaged loss
#     """
#     B, C, H, W = logits.shape
#     probs = torch.softmax(logits, dim=1)
#     prediction = torch.argmax(probs, dim=1).long().cpu()
#     valid = (target != 0)
#     target_without_invalids = (target-1) * valid

#     imgs = []
#     gts = []
#     for img in prediction:
#         image =  flooding_model.rgb_for_Canny(img)
#         logits_c = Canny(image)
#         imgs.append(logits_c//255)

#     for gt in target_without_invalids:
#         mask = flooding_model.rgb_for_Canny(gt)
#         target_c = Canny(mask)
#         gts.append(target_c//255)

#     imgs = np.array(imgs)
#     logits = torch.tensor(imgs).to(logits.device).float()
#     gts = np.array(gts)
#     target = torch.tensor(gts).to(logits.device).float()
    
#     distance_matrix = torch.cdist(logits, target, p=p)  # p=2 means Euclidean Distance
#     distance_matrix = distance_matrix/torch.max(distance_matrix+1e-8)

#     value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
#     value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

#     value = torch.cat((value1, value2), dim=1)
#     loss = value.max(1)[0].mean()
#     # print(loss)
    
#     return loss


def calc_loss_fc_iou_bcec(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """
    w_fc = nn.Parameter(torch.tensor(bce_weight))
    w_iou = nn.Parameter(torch.tensor(bce_weight))
    w_C = nn.Parameter(torch.tensor(bce_weight))
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=5, alpha=0.01).requires_grad_()
    # iou = iou_loss_mask_invalid(logits, target) # (B, C)
    # bce_c = BCE_Canny_loss_mask_invalid(logits, target)
    # fc_c = Fc_Canny_loss_mask_invalid(logits, target)
    H_c = Hausdorff_Canny_loss_mask_invalid(logits, target).requires_grad_()
    # Haus = HausdorffLoss(logits, target)
    # Weighted sum
    return fc * w_fc + H_c * w_C

                               