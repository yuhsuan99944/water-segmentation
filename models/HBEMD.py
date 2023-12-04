from __future__ import division, print_function

import logging
import numpy as np
import torch
import cv2
import torch.nn as nn
from scipy.interpolate import Rbf
import pylab as plt
# import torchplot as tp
import torchvision.transforms as T
from torch.nn import functional as F

try:
#     from sk.greyreconstruct import reconstruction
    from skimage.morphology import reconstruction
except ImportError:
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BEMD:
    logger = logging.getLogger(__name__)

    def __init__(self):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.FIXE = 1  # Single iteration by default, otherwise results are terrible
        self.FIXE_H = 0
        self.MAX_ITERATION = 5

    def __call__(self, image, max_imf=-1):
        return self.bemd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image, min_peaks_pos, max_peaks_pos):
        """Calculates top and bottom envelopes for image.
        Parameters
        ----------
        image : numpy 2D array
        Returns
        -------
        min_env : numpy 2D array
            Bottom envelope in form of an image.
        max_env : numpy 2D array
            Top envelope in form of an image.
        """
        # torch  #(8,256,256)
        # xi, yi = torch.meshgrid(torch.arange(image.shape[0]), torch.arange(image.shape[1]))
        # min_val = torch.tensor([image[z-1][x,y] for x, y, z in zip(*min_peaks_pos)])  # min_peaks_pos:(tensor(0~255), tensor(0~255), tensor(0~7))
        # max_val = torch.tensor([image[z-1][x,y] for x, y, z in zip(*max_peaks_pos)])
        # min_env = self.spline_points(min_peaks_pos[0], min_peaks_pos[1], min_val, xi, yi)
        # max_env = self.spline_points(max_peaks_pos[0], max_peaks_pos[1], max_val, xi, yi)
        
        min_peaks_pos = min_peaks_pos.unsqueeze(1)
        # print("min_peaks_pos",min_peaks_pos.device)
        min_env = F.interpolate(min_peaks_pos,size=256,mode='bicubic',align_corners=True).squeeze()
        # print("min_env",min_env.device)
        max_peaks_pos = max_peaks_pos.unsqueeze(1)
        # print("max_peaks_pos",max_peaks_pos.device)
        max_env = F.interpolate(max_peaks_pos,size=256,mode='bicubic',align_corners=True).squeeze()
        # print("max_env",max_env.device)
        return min_env, max_env     #min_env, max_env

#     @classmethod
#     def spline_points(cls, X, Y, Z, xi, yi):
#         """Creates a spline for given set of points.
#         Uses Radial-basis function to extrapolate surfaces. It's not the best but gives something.
#         Griddata algorithm didn't work.
#         """
#         # spline = Rbf(X, Y, Z, function='cubic')
#         Rbf = RBF_2D()
#         F = Rbf.RBF(X, Y, Z, xi, yi)
        
#         return F  #spline(xi, yi)

    @classmethod
    def find_extrema_positions(cls, image):
        """
        Finds extrema, both mininma and maxima, based on morphological reconstruction.
        Returns extrema where the first and second elements are x and y positions, respectively.
        Parameters
        ----------
        image : numpy 2D array
            Monochromatic image or any 2D array.
        Returns
        -------
        min_peaks_pos : numpy array
            Minima positions.
        max_peaks_pos : numpy array
            Maxima positions.
        """
        # print("image",image.device)
        max_peaks_pos = BEMD.extract_maxima_positions(image)
        # print("max_peaks_pos",max_peaks_pos.device)
        min_peaks_pos = BEMD.extract_minima_positions(image)
        # print("min_peaks_pos",min_peaks_pos.device)
        return min_peaks_pos, max_peaks_pos

    @classmethod
    def extract_minima_positions(cls, image):
        return BEMD.extract_maxima_positions(-image)

    @classmethod
    def extract_maxima_positions(cls, image):
        # seed_min = image - 1
        seed_min = image - 0.000001 # case on sin and cos of image
        max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        tensor_dilate = max_pool(image.unsqueeze(0))  # (1,8,256,256)
        # print("tensor_dilate",tensor_dilate.device)
        cleaned_image = image - tensor_dilate.squeeze()
        # print("cleaned_image",cleaned_image.device)
        maxima_positions = torch.where(cleaned_image<0)[::-1]

        return cleaned_image # maxima_positions  # tp.imshow(cleaned_image.squeeze()) #.shape      #show(tensor_dilate).show()

    @classmethod
    def end_condition(cls, image, IMFs):
        """Determins whether decomposition should be stopped.
        Parameters
        ----------
        image : numpy 2D array
            Input image which is decomposed.
        IMFs : numpy 3D array
            Array for which first dimensions relates to respective IMF,
            i.e. (numIMFs, imageX, imageY).
        """
        # torch
        rec = torch.sum(IMFs, dim=0).float()
        # print("rec",rec.device)
        # If reconstruction is perfect, no need for more tests
        if torch.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        """Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.
        Parameters
        ----------
        proto_imf : numpy 2D array
            Current iteration of proto IMF.
        proto_imf_prev : numpy 2D array
            Previous iteration of proto IMF.
        mean_env : numpy 2D array
            Local mean computed from top and bottom envelopes.
        Returns
        -------
        boolean
            Whether current proto IMF is actual IMF.
        """
        #TODO: Sifiting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decompoisition and thus repeating above/below
        #      behaviour. For now, mean_env is checked whether close to zero excluding
        #      its offset.
        # print("proto_imf",proto_imf.device)
        # print("proto_imf_prev",proto_imf_prev.device)
        # print("mean_env",mean_env.device)
        
        # torch
        if torch.all(torch.abs(mean_env-mean_env.mean())<self.mean_thr):
        #if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if torch.allclose(proto_imf, proto_imf_prev, rtol=0.01):
            return True

        # If IMF mean close to zero (below threshold)
        if torch.mean(torch.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = torch.mean(proto_imf*proto_imf)
        if mse_proto_imf > self.mse_thr:
            return False

        return False

    def bemd(self, image, max_imf=-1):
        """Performs bidimensional EMD (BEMD) on grey-scale image with specified parameters.
        Parameters
        ----------
        image : numpy 2D array,
            Grey-scale image.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.
        Returns
        -------
        IMFs : numpy 3D array
            Set of IMFs in form of numpy array where the first dimension
            relates to IMF's ordinary number.
        """
        # torch
         
        # image = np.array(image.cpu()) #.numpy()  # (8,1,256,256) > numpy
        image = image.squeeze()#.cuda()
        image_s = image.clone().detach()
        # print("image_s",image_s.device)
        imf = torch.zeros(image.shape).to(device_)
        imf_old = imf.clone()
        # print("imf_old",imf_old.device)
        imfNo = 0
        IMF = torch.empty((imfNo,)+image.shape).squeeze().to(device_)   # (0,8,256,256)
        # print("IMF",IMF.device)
        notFinished = True

        while(notFinished):
            # self.logger.debug('IMF -- '+str(imfNo))
            # self.logger.debug('Shape IMF: ',IMF.shape)
            # self.logger.debug('IMF: ',IMF[:imfNo])
            res = image_s - torch.sum(IMF[:imfNo], dim=0) # (8,8,256,256)
            
            # saveLogFile('residue_' + str(imfNo) + '.csv',res)
            imf = res.clone()
            # print("imf",imf.device)
            mean_env = torch.zeros(image.shape).to(device_)
            stop_sifting = False

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when mean(proto_imf) < threshold

            while(not stop_sifting and n<self.MAX_ITERATION):
                n += 1
                # self.logger.debug("Iteration: %i", n)

                min_peaks_pos, max_peaks_pos = self.find_extrema_positions(imf)

                # self.logger.debug("min_peaks_pos = %i  |  max_peaks_pos = %i", len(min_peaks_pos[0]), len(max_peaks_pos[0]))
                if len(min_peaks_pos[0])>1 and len(max_peaks_pos[0])>1:
                    min_env, max_env = self.extract_max_min_spline(imf, min_peaks_pos, max_peaks_pos)
                    mean_env = 0.5*(min_env+max_env)
                    imf_old = imf.clone()
                    # print("imf_old",imf_old.device)
                    imf = imf - mean_env
                    # Fix number of iterations
                    if self.FIXE:
                        if n>=self.FIXE+1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        if n == 1: continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    stop_sifting = True

            IMFIMF = torch.vstack((IMF, imf.clone()[None,:]))
            # print("IMFIMF",IMFIMF.device)
            imfNo += 1

            if self.end_condition(image, IMF) or (max_imf>0 and imfNo>=max_imf):
                notFinished = False
                break

        res = image_s - torch.sum(IMF[:imfNo], dim=0).float()
        zero = torch.zeros(res.shape).to(device_)
        if not torch.allclose(res, zero):
            IMF = torch.vstack((IMF, res[None,:]))
            # print("IMF",IMF.device)
            imfNo += 1
        return IMF
    
def changeNegativeHueToPositive(hue_channel):
    for i in range(0,hue_channel.shape[0]):
        for j in range(0,hue_channel.shape[1]):
            for k in range(0, hue_channel.shape[2]):
                if(hue_channel[i][j][k] < 0):
                    hue_channel[i][j][k] = hue_channel[i][j][k] + 360
    return hue_channel

class HBEMD():
    def rgb2hsv_torch(self, rgb: torch.Tensor) -> torch.Tensor:
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
        # print("hsv_h", hsv_h.device)
        # hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        # hsv_v = cmax
        #torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
        return hsv_h

    def sifting(self,hue_img, max_imf_value):
        # torch
        # hue_img = torch.tensor(hue_img)  # , dtype = torch.float32
        hueradians = torch.deg2rad(hue_img)
        cos_hueradians = torch.cos(hueradians)
        sin_hueradians = torch.sin(hueradians)

        sin_bemd = BEMD()
        imfs_sin_hue = sin_bemd.bemd(sin_hueradians, max_imf=max_imf_value)
        cos_bemd = BEMD()
        imfs_cos_hue = cos_bemd.bemd(cos_hueradians, max_imf=max_imf_value)

#         h_bemd_imfs = []
        imfs_no = min(imfs_sin_hue.shape[0], imfs_cos_hue.shape[0])
        for i in range(0, imfs_no):
            imf_cos_hue = imfs_cos_hue[i]
            imf_sin_hue = imfs_sin_hue[i]
            imf_arctan_hue = torch.atan2(imf_sin_hue, imf_cos_hue)
            imf_hue_degree = torch.rad2deg(imf_arctan_hue)
            imf_hue_degree_old = torch.clone(imf_hue_degree)
            imf_hue_degree = changeNegativeHueToPositive(imf_hue_degree_old)
#             h_bemd_imfs.append(imf_hue_degree)

        return  imf_hue_degree # h_bemd_imfs

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HBEMD_F(nn.Module):
    def __init__(self, img, feature):
        super(HBEMD_F, self).__init__()
        self.hbemd = nn.Conv2d(img, 1, kernel_size=1).to(device_)
        
    def forward(self, x):
        h_bemd = HBEMD()
        # hue_img = h_bemd.hue_value()
        hue_img = h_bemd.rgb2hsv_torch(x).to(device_)
        # print("hue_img", hue_img.device)
        # imfs = h_bemd.sifting(hue_img, max_imf_value=2).unsqueeze(1).to(device_) #.cuda()#.unsqueeze(0)
        imfs = hue_img.to(device_)
        # print("imfs",imfs.shape)
        H = self.hbemd(imfs)
        # C1 = torch.load(H)
        return H
    
# class Hue():
    