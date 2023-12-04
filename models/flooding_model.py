import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple
from ml4floods.preprocess.worldfloods import normalize
from ml4floods.models.config_setup import  AttrDict
from pytorch_lightning.utilities.cloud_io import load

from ml4floods.models.utils import metrics
# from ml4floods.models.architectures.baselines import SimpleLinear, SimpleCNN
from ml4floods.models.architectures.baselines import SimpleLinear
# from ml4floods.models.architectures.unets import UNet, UNet_dropout
from ml4floods.models.architectures.hrnet_seg import HighResolutionNet
from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS, CHANNELS_CONFIGURATIONS, BANDS_S2, COLORS_WORLDFLOODS_INVLANDWATER, COLORS_WORLDFLOODS_INVCLEARCLOUD
from ml4.data.worldfloods.configs import COLORS_WORLDFLOODS_LWC

from models.architecture import SimpleCNN, ViT, RGB2HSV
from models.unet_optimize import UNet, UNet_dropout, SimpleUNet, FullUNet, UNET2, Res2_UNET, Res2_DAUNET, AttUNET, Res2_AttUNET, DAUNET, Res2_SAUNET, RDN_Res2_AttUNET
from models.unet_optimize import UNET2, Res2_UNET, Res2_DAUNET, AttUNET, Res2_AttUNET, Res2_AttUNET_Sup, Simple_Res2Unet, HUNet
from models.unet_optimize import TransUNet, Res2VTUnet#, MSVTUnet #CvTUnet, 
from models.Compare_model import UTNet #, SwinTransformerSys, DeepLab_V3_plus,MTUNet,    #MALUNet

from models import losses
from torch import nn
import torchvision.transforms.functional as TF
from models.adach import AdaCh

class WorldFloodsModel0(pl.LightningModule): # bce
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid_original_unet(logits, y, weight=self.weight_per_class.to(self.device))
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        
        # bce_loss = losses.cross_entropy_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device))
        # dice_loss = losses.dice_loss_mask_invalid(logits, y)
        
        bce_loss = losses.calc_loss_mask_invalid_original_unet(logits, y, weight=self.weight_per_class.to(self.device))
        self.log('val_bce_loss', bce_loss)
        # self.log('val_dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }
    

class WorldFloodsModel(pl.LightningModule): #bce+dice
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device))
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        
        bce_loss = losses.cross_entropy_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device))
        dice_loss = losses.dice_loss_mask_invalid(logits, y)
        self.log('val_bce_loss', bce_loss)
        self.log('val_dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }

class WorldFloodsModel1(pl.LightningModule): # fc+dice
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid_2(logits, y, weight=self.weight_per_class.to(self.device))
        
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        
        focal_loss = losses.focal_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device), gamma=5, alpha=0.001)
        dice_loss = losses.dice_loss_mask_invalid(logits, y)
        self.log('val_focal_loss', focal_loss)
        self.log('val_dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }

class WorldFloodsModel2(pl.LightningModule): # fc+iou
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        self.reg_lambda = 0.01
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        
        logits = self.network(x)
        
        has_water = (gt == 2).any(dim=-1).any(dim=-1).squeeze(1)
        logits = logits * (has_water).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        y = y *  (has_water).unsqueeze(-1).unsqueeze(-1)
       
        # loss = losses.calc_loss_mask_invalid_3(logits, y, weight=self.weight_per_class.to(self.device))
        loss = losses.calc_loss_fc_iou_bcec(logits, y, weight=self.weight_per_class.to(self.device))
        if self.reg_lambda > 0:
            regularization_loss = 0
            for param in self.parameters():
                regularization_loss += torch.norm(param, p=2)
            loss += self.reg_lambda * regularization_loss
        
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
            # self.log("ir", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        # torch.cuda.empty_cache()
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        logits = self.network(x)
        
        has_water = (gt == 2).any(dim=-1).any(dim=-1).squeeze(1)
        logits = logits * (has_water).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        y = y *  (has_water).unsqueeze(-1).unsqueeze(-1)
        # focal_loss = losses.focal_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device), gamma=5, alpha=0.01)
        
        focal_loss = losses.focal_loss_mask_invalid(logits, y, gamma=5, alpha=0.01)
        iou_loss = losses.iou_loss_mask_invalid(logits, y)
        # fc_C_loss = losses.Fc_Canny_loss_mask_invalid(logits, y)
        # bceC_loss = losses.BCE_Canny_loss_mask_invalid(logits, y)
        H_C_loss = losses.Hausdorff_Canny_loss_mask_invalid(logits,y)
        # haus_loss = losses.HausdorffLoss(logits,y)
        
        
        self.log('val_focal_loss', focal_loss)
        self.log('val_iou_loss', iou_loss)
        # self.log('val_bceC_loss', bceC_loss)
        self.log('val_hausC_loss', H_C_loss)
        # self.log('val_fcC_loss', fc_C_loss)
        # self.log('val_haus_loss', haus_loss)
        

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }
    
    
class WorldFloodsModel3(pl.LightningModule): # fc+iou+ssim
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid_4(logits, y, weight=self.weight_per_class.to(self.device))
        # print(loss)
        # print(type(loss))
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        logits = self.network(x)
        
        # focal_loss = losses.focal_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device), gamma=2, alpha=0.25)
        iou_loss = losses.iou_loss_mask_invalid(logits, y)
        # dice_loss = losses.dice_loss_mask_invalid(logits, y)
        ssim_loss = losses.ssim(logits, y)
        
        # self.log('val_focal_loss', focal_loss)
        self.log('val_iou_loss', iou_loss)
        # self.log('val_dice_loss', dice_loss)
        self.log('val_ssim_loss', ssim_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }
    
class WorldFloodsModel_AdaCh(pl.LightningModule): # fc+iou
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.gt = None
        # gt = gt
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        # self.network = configure_architecture_gt(h_params_dict, gt)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)   #训练过程中每次更新权重时的步长，它的大小会影响训练速度和模型性能
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)  #在训练过程中每次更新学习率的系数                               # ori: 0.5
        self.lr_patience = h_params_dict.get('lr_patience', 2)  #在训练过程中如果模型的性能没有得到显著提高，就降低学习率     # ori: 2
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])
        self.adach = AdaCh()

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        
        # print("x", x.shape)
        # print("gt", gt.shape, gt)
        RGBIH_AdaCh = self.adach(x, gt) #(B, 5, 256, 256)
        logits = self.network(RGBIH_AdaCh)
        
        has_water = (gt == 2).any(dim=-1).any(dim=-1).squeeze(1)
        logits = logits * (has_water).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        y = y.float() *  (has_water).unsqueeze(-1).unsqueeze(-1)
       
        loss = losses.calc_loss_mask_invalid_3(logits, y)
        # loss = losses.calc_loss_mask_invalid_3(logits, y, weight=self.weight_per_class.to(self.device))
        # 打印梯度值
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
                
        if (batch_idx % 100) == 0:
            self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        # torch.cuda.empty_cache()
        return self.network(x)

    def log_images(self, x, y, logits,prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        # print("x", x.shape)
        # print("gt", gt.shape)
        RGBIH_AdaCh = self.adach(x, gt)
        logits = self.network(RGBIH_AdaCh)
        
        # focal_loss = losses.focal_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device), gamma=5, alpha=0.01)
        focal_loss = losses.focal_loss_mask_invalid(logits, y, gamma=5, alpha=0.01)
        iou_loss = losses.iou_loss_mask_invalid(logits, y)
        self.log('val_focal_loss', focal_loss)
        self.log('val_iou_loss', iou_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits,prefix="val_")
            # ir = x[:, 3, :, :]
            # self.log_images(x, y, ir,prefix="val_")
            # h = x[:, 4, :, :]
            # self.log_images(x, y, h,prefix="val_")
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }
    

class WorldFloodsModel_Sup(pl.LightningModule): # fc+iou+ssim
    """
    Model to do multiclass classification.
    It expects ground truths y (B, H, W) tensors to be encoded as: {0: invalid, 1: clear, 2:water, 3: cloud}
    The preds (model.forward(x)) will produce a tensor with shape (B, 3, H, W)
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = configure_architecture(h_params_dict)
        self.weight_per_class = torch.Tensor(h_params_dict.get('weight_per_class',
                                                               [1 for i in range(self.num_class)]),
                                             device=self.device)
        # self.batch_size = data_params.get('batch_size')
        self.max_epoch = h_params_dict.get('max_epochs')
        
        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-5)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 1)
        self.reg_lambda = 0.01
        self.boundary_executed = False
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])
        self._fc_val = 0.5
        self._iou_lss_val = 0.5

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        logits = self.network(x)

        has_water = (gt == 2).any(dim=-1).any(dim=-1).squeeze(1)
        logits = logits * (has_water).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        y = y *  (has_water).unsqueeze(-1).unsqueeze(-1)

        loss= losses.calc_loss_dice_biou(logits, y, weight=self.weight_per_class.to(self.device))

###### Part Loss
        # if (self.trainer.logged_metrics.get('val_iou_loss', 0.3) > 0.293 and not self.boundary_executed):
        # loss = losses.calc_loss_mask_invalid_3(logits, y)
        #     self.boundary_executed = False
        # else :
        #     loss = losses.Hausdorff_Canny_loss_mask_invalid(logits, y)   
        #     self.boundary_executed = True

        # # loss = losses.Hausdorff_Canny_loss_mask_invalid(logits, y) 
        # # print(loss)
        # if self.reg_lambda > 0:
        #     regularization_loss = 0
        #     for param in self.parameters():
        #         regularization_loss += torch.norm(param, p=2)
        #     # loss += self.reg_lambda * regularization_loss
        #     loss = torch.add(loss, self.reg_lambda * regularization_loss)

            
            
###### Supvision           
#         for ix in all_layer:
#             loss = losses.calc_loss_mask_invalid_3(ix, y, weight=self.weight_per_class.to(self.device))
#             # boundary_loss = losses.Hausdorff_Canny_loss_mask_invalid(ix, y)                          
#             all_loss.append(boundary_loss)
#         edge_loss = sum(all_loss) / self.batch_size  # ori:nAvg* bs
       
#         compound_loss = loss + edge_loss
        


        if (batch_idx % 100) == 0:
            self.log("loss", loss)
            # self.log("tra_weight_biou", w_biou)
            # self.log("tra_weight_dice", w_dice)
            

        # 圖片暫時為最後輸出
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits, prefix="train_")      

        return loss   # 自動更新
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor
        Returns:
            (B, 3, H, W) prediction of the network
        """
        return self.network(x)

    def log_images(self, x, y, logits, prefix=""):
        import wandb
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = batch_to_unnorm_rgb(x,
                                       self.hparams["model_params"]["hyperparameters"]['channel_configuration'])

        self.logger.experiment.log(
            {f"{prefix}overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({f"{prefix}y": [wandb.Image(mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({f"{prefix}pred": [wandb.Image(mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze(1)
        gt = batch['mask']
        logits = self.network(x)

        has_water = (gt == 2).any(dim=-1).any(dim=-1).squeeze(1)
        logits = logits * (has_water).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        y = y *  (has_water).unsqueeze(-1).unsqueeze(-1)
        
        
        # all_boundary = []
        # all_fc = []
        # all_iou = []
        # boundary_loss = losses.Hausdorff_Canny_loss_mask_invalid(logits, y)
        # focal_loss = losses.focal_loss_mask_invalid(logits, y, weight=self.weight_per_class.to(self.device), gamma=5, alpha=0.01)
        # iou_loss = losses.iou_loss_mask_invalid(logits, y)
        dice_loss= losses.dice_loss_mask_invalid(logits, y)
        biou_loss= losses.Boundary_iou_loss_mask_invalid(logits, y) 
        
#         for ix in all_layer:
#             focal_loss = losses.focal_loss_mask_invalid(ix, y, weight=self.weight_per_class.to(self.device), gamma=5, alpha=0.01)
#             iou_loss = losses.iou_loss_mask_invalid(ix, y)
#             all_fc.append(focal_loss)
#             all_iou.append(iou_loss)
#             # all_boundary.append(boundary_loss)
          
#         # boundary_loss = min(all_boundary)
#         focal_loss = min(all_fc)
#         iou_loss = min(all_iou)

        # self.log('val_focal_loss', focal_loss)
        # self.log('val_iou_loss', iou_loss)
        self.log('val_dice_loss', dice_loss)
        self.log('val_biou_loss', biou_loss)
        # self.log('val_boundary_loss', bound
        
        pred_categorical = torch.argmax(logits, dim=1).long()  # 返回指定维度最大值的序号
        # print(pred_categorical.shape)

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class, remove_class_zero=True)
        # print(cm_batch.shape)

        # Log accuracy per class
        recall = metrics.calculate_recall(cm_batch, self.label_names)
        for k in recall.keys():
            self.log(f"val_recall {k}", recall[k])

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"val_iou {k}", iou_dict[k])
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits, prefix="val_")


    def configure_optimizers(self):
        # if (self.trainer.logged_metrics.get('val_iou_loss', 0.3) > 0.293 and not self.boundary_executed):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)
        #     self.boundary_executed = False
        # else:
        #     self.lr = 0.0001  # 改变 self.lr 的值
        #     optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                            factor=self.lr_decay, verbose=True,
        #                                                            patience=self.lr_patience)
        #     self.boundary_executed = True
        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}

    def wb_mask(self, bg_img, pred_mask, true_mask):
        import wandb
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }
    
    

class DistilledTrainingModel(WorldFloodsModel):
    def __init__(self, teacher_model_params: dict, student_model_params: dict):
        super().__init__(model_params=student_model_params)
#         super().__init__(model_params=teacher_model_params)
#         self.save_hyperparameters()
                
        teacher_params_dict = teacher_model_params.get('hyperparameters', {})
        self.num_class = teacher_params_dict.get('num_classes', 3)
        self._teacher_network = configure_architecture(teacher_params_dict)
        
        student_params_dict = student_model_params.get('hyperparameters', {})
        self._student_network = configure_architecture(student_params_dict)
  
        #label names setup
        self.label_names = student_model_params.get('label_names', [i for i in range(self.num_class)])
        self._mse_loss = nn.MSELoss()

    def training_step(self, batch: Dict, batch_idx) -> float:
        x, y = batch['image'], batch['mask'].squeeze(1)
        y_logits_student = self.forward(x)
        y_logits_teacher = self._teacher_network.forward(x)
        loss = self._mse_loss(y_logits_student, y_logits_teacher)
        return loss
#         return {'loss': loss,
#                 'log': {'train_loss': loss}}

# def configure_architecture_gt(h_params:AttrDict, gt: Dict) -> torch.nn.Module:
#     architecture = h_params.get('model_type', 'linear')
#     num_channels = h_params.get('num_channels', 3)
#     num_classes = h_params.get('num_classes', 2)
#     # ground_truth = h_params.get('')
    
#     if architecture == "hunet":
#         model = HUNet(num_channels, num_classes)
        
#     else:
#         raise Exception(f'No model implemented for model_type: {h_params.model_type}')

#     return model
    
    
def configure_architecture(h_params:AttrDict) -> torch.nn.Module:
    architecture = h_params.get('model_type', 'linear')
    num_channels = h_params.get('num_channels', 3)
    num_classes = h_params.get('num_classes', 2)

    if architecture == 'unet':
        print('num of channels: ', num_channels, ', num of classes: ', num_classes)
        model = UNet(num_channels, num_classes)
    
    elif architecture == 'unet_simple':
        model = SimpleUNet(num_channels, num_classes)
        
    elif architecture == 'full_unet':
        model = FullUNet(num_channels, num_classes)

    elif architecture == 'simplecnn':
        model = SimpleCNN(num_channels, num_classes)

    elif architecture == 'linear':
        model = SimpleLinear(num_channels, num_classes)

    elif architecture == 'unet_dropout':
        model = UNet_dropout(num_channels, num_classes)

    elif architecture == "hrnet_small":
        model = HighResolutionNet(input_channels=num_channels, output_channels=num_classes)
        
        if num_channels == 3:
            print("3-channel model. Loading pre-trained weights from ImageNet")
            # TODO models are bgr instead of rgb!
            pretrained_dict = load(PATH_TO_MODEL_HRNET_SMALL)
            model.init_weights(pretrained_dict)

    elif architecture == "unet2":
        model = UNET2(num_channels, num_classes)
        
    elif architecture == "hunet":
        model = HUNet(num_channels, num_classes)
    #####
#     elif architecture == "segnet":
#         model = SegNet(num_channels, num_classes)
        
#     elif architecture == "deeplabv3":
#         model = DeepLabV3(num_channels, num_classes)

    elif architecture == "malunet":
        model = MALUNet(num_channels, num_classes)
        
    elif architecture == "transunet":
        model = TransUNet(num_channels, num_classes)
        
    elif architecture == "res2vtunet":
        model = Res2VTUnet(num_channels, num_classes)
        
    # elif architecture == "msvtunet":
    #     model = MSVTUnet(num_channels, num_classes)
    
    elif architecture == "vit":
        model = ViT(num_channels, num_classes)
    #####
        
    elif architecture == "res2_unet":
        model = Res2_UNET(num_channels, num_classes)
        
    elif architecture == "attunet":
        model = AttUNET(num_channels, num_classes)
        
    elif architecture == "res2_attunet":
        model = Res2_AttUNET(num_channels, num_classes)
        
    # elif architecture == "swinunet":
    #     model = SwinTransformerSys(num_channels, num_classes)
        
    # elif architecture == "deeplabv3+":
    #     model = DeepLab_V3_plus(num_channels, num_classes)

    # elif architecture == "mtunet":
    #     model = MTUNet(num_channels, num_classes)
    elif architecture == "utnet":
        model = UTNet(num_channels, num_classes)
        
    elif architecture == 'res2_attunet_sup':
        model = Res2_AttUNET_Sup(num_channels, num_classes)
        
    elif architecture == 'simp_res2unet':
        model = Simple_Res2Unet(num_channels, num_classes)
        
    else:
        raise Exception(f'No model implemented for model_type: {h_params.model_type}')

    return model    

def mask_to_rgb(mask, values=[0, 1, 2, 3], colors_cmap=COLORS_WORLDFLOODS):
    """
    Given a 2D mask it assign each value of the mask the corresponding color
    :param mask:
    :param values:
    :param colors_cmap:
    :return:
    """
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"
    assert len(mask.shape) == 2, f"Unexpected shape {mask.shape}"
    mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c
    return mask_return

def rgb_for_Canny(mask, values=[0, 1, 2], colors_cmap=COLORS_WORLDFLOODS_LWC):  
    #{0:Land, 1:water, 2:Cloud}
    """
    Given a 2D mask it assign each value of the mask the corresponding color
    :param mask:
    :param values:
    :param colors_cmap:
    :return:
    """
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"
    assert len(mask.shape) == 2, f"Unexpected shape {mask.shape}"
    mask = mask.cpu()
    mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c
    return mask_return

# def batch_to_unnorm_rgb(x:torch.Tensor, channel_configuration:str="all", max_clip_val=3000.) -> np.ndarray:
#     """
#     Unnorm x images and get rgb channels for visualization

#     Args:
#         x: (B, C, H, W) image
#         channel_configuration: one of CHANNELS_CONFIGURATIONS.keys()
#         max_clip_val: value to saturate the image

#     Returns:
#         (B, H, W, 3) np.array with values between 0-1 ready to be used in imshow/PIL.from_array()
#     """
#     model_input_npy = x.cpu().numpy()

#     mean, std = normalize.get_normalisation("bgr")  # B, R, G!
#     mean = mean[np.newaxis]
#     std = std[np.newaxis]

#     # Find the RGB indexes within the S2 bands
#     bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
#     bands_index_rgb = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]

#     model_input_rgb_npy = model_input_npy[:, bands_index_rgb].transpose(0, 2, 3, 1) * std[..., -1::-1] + mean[...,
#                                                                                                          -1::-1]
#     model_input_rgb_npy = np.clip(model_input_rgb_npy / max_clip_val, 0., 1.)

#     return model_input_rgb_npy
def batch_to_unnorm_rgb(x:torch.Tensor, channel_configuration:str="all", max_clip_val=3000.,
                        unnormalize:bool=True) -> np.ndarray:
    """
    Unnorm x images and get rgb channels for visualization
    Args:
        x: (B, C, H, W) image
        channel_configuration: one of CHANNELS_CONFIGURATIONS.keys()
        max_clip_val: value to saturate the image
        unnormalize:
    Returns:
        (B, H, W, 3) np.array with values between 0-1 ready to be used in imshow/PIL.from_array()
    """
#     model_input_npy = x.cpu().numpy()
#     model_input_npy = x[:,:-1, :, :]
#     # print(model_input_npy.shape)

#     mean, std = normalize.get_normalisation("rgb")  # B, R, G!
#     mean = mean[np.newaxis]
#     # # print(mean)
#     # print(mean.shape)
#     std = std[np.newaxis]
#     # print(std)
#     # print(std.shape)

#     # Find the RGB indexes within the S2 bands
#     bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
#     bands_index_rgb = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]
#     print(bands_index_rgb)
#     print(type(bands_index_rgb))
#     print(model_input_npy[:, bands_index_rgb].shape)
#     model_input_rgb_npy = model_input_npy[:, bands_index_rgb].transpose(0, 2, 3, 1)
    
#     if unnormalize:
#         model_input_rgb_npy = model_input_npy  * std + mean
#         model_input_rgb_npy = np.clip(model_input_rgb_npy / max_clip_val, 0., 1.)

    model_input_npy = x.cpu().numpy()

    mean, std = normalize.get_normalisation("bgr")  # B, R, G!
    mean = mean[np.newaxis]
    std = std[np.newaxis]

    # Find the RGB indexes within the S2 bands
    bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands_index_rgb = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]

    model_input_rgb_npy = model_input_npy[:, bands_index_rgb].transpose(0, 2, 3, 1) * std[..., -1::-1] + mean[...,
                                                                                                         -1::-1]
    model_input_rgb_npy = np.clip(model_input_rgb_npy / max_clip_val, 0., 1.)

    return model_input_rgb_npy


    

def batch_mask_to_rgb(x:torch.Tensor):
    """
    Args:
        x: (B, 1, H, W) image

    Returns:
        (B, H, W, 1) np.array with values between 0-1 ready to be used in imshow/PIL.from_array()
    """
    model_input_npy = x.cpu().numpy()
    # print(model_input_npy.shape)
    model_input_npy = np.transpose(model_input_npy, (0, 2, 3, 1))
    # print(model_input_npy.shape)
    return model_input_npy
    

def unnorm_batch(x:torch.Tensor, channel_configuration:str="all", max_clip_val:float=3000.) ->torch.Tensor:
    model_input_npy = x.cpu().numpy()

    mean, std = normalize.get_normalisation(channel_configuration, channels_first=True)
    # print(mean.shape)
    mean = mean[np.newaxis] # (1, nchannels, 1, 1)
    std = std[np.newaxis]  # (1, nchannels, 1, 1)
    out = model_input_npy * std + mean
    if max_clip_val is not None:
        out = np.clip(out/max_clip_val, 0, 1)
    return out


def plot_batch(x:torch.Tensor, channel_configuration:str="all", bands_show=None, axs=None, max_clip_val=3000.,
               show_axis=False):
    """

    Args:
        x:
        channel_configuration:
        bands_show: RGB ["B4", "B3", "B2"] SWIR/NIR/RED ["B11", "B8", "B4"]
        axs:
        max_clip_val: value to saturate the image
        show_axis: Whether to show axis of the image or not

    Returns:

    """
    import matplotlib.pyplot as plt

    if bands_show is None:
        bands_show = ["B4", "B3", "B2"]

    if axs is None:
        fig, axs = plt.subplots(1, len(x))

    x = unnorm_batch(x, channel_configuration, max_clip_val=max_clip_val)
    bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands_index_rgb = [bands_read_names.index(b) for b in bands_show]
    x = x[:, bands_index_rgb]

    if hasattr(x, "cpu"):  #判断对象是否包含对应的属性
        x = x.cpu()

    for xi, ax in zip(x, axs):
        xi = np.transpose(xi, (1, 2, 0))
        ax.imshow(xi)
        if not show_axis:
            ax.axis("off")
    # print(xi.shape)
def read_batch(x:torch.Tensor, channel_configuration:str="all", bands_show=None, axs=None, max_clip_val=3000.,
               show_axis=False):
   

    if bands_show is None:
        bands_show = ["B4", "B3", "B2"]

    x = unnorm_batch(x, channel_configuration, max_clip_val=max_clip_val)
    bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands_index_rgb = [bands_read_names.index(b) for b in bands_show]
    x = x[:, bands_index_rgb]  # 只要RGB的band

    if hasattr(x, "cpu"):
        x = x.cpu()
    # print(x)
        
    # new_x = [8]
    for xi in x:  # 兩個 list 以迴圈的方式一次各取一個元素出來處理，可以使用 zip 打包之後配合 for 迴圈來處理 [8,3,572,572]
        xi = np.transpose(xi,(1, 2, 0)) # change [3,572,572] --> [572,572,3]
        #global nex_x
    
    # new_x.append(xi) 
    # print(new_x)
    return xi # [8, 572, 572, 3]

def plot_batch_output_v1(outputv1: torch.Tensor, axs=None, legend=True, show_axis=False):
    """

    Args:
        outputv1:  (B, W, H) Tensor encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        axs:
        legend: whether to show the legend or not
        show_axis:  Whether to show axis of the image or not

    Returns:

    """
    import matplotlib.pyplot as plt
    from ml4floods.visualization import plot_utils

    if hasattr(outputv1, "cpu"):
        outputv1 = outputv1.cpu()

    if axs is None:
        axs = plt.subplots(1, len(outputv1))

    cmap_preds, norm_preds, patches_preds = plot_utils.get_cmap_norm_colors(plot_utils.COLORS_WORLDFLOODS,
                                                                            plot_utils.INTERPRETATION_WORLDFLOODS)

    for _i, (xi, ax) in enumerate(zip(outputv1, axs)):
        ax.imshow(xi, cmap=cmap_preds, norm=norm_preds,
                  interpolation='nearest')

        if not show_axis:
            ax.axis("off")

        if _i == (len(outputv1)-1) and legend:
            ax.legend(handles=patches_preds,
                      loc='upper right')
    # print(xi.shape)