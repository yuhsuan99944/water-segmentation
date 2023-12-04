import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt

from models import flooding_model
from ml4floods.models.config_setup import get_default_config
from pytorch_lightning.utilities.cloud_io import load
from models.flooding_model import WorldFloodsModel_Sup
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.visualization import plot_utils
from ml4floods.data.worldfloods import dataset


config_fp = "/home/viplab/VipLabProjects/yuyu/train_models/last_run/config_rgbnirh_worldfloodsup_res2vtunet_ep30.json"
config = get_default_config(config_fp)

importlib.reload(flooding_model)
model = WorldFloodsModel_Sup(config.model_params, config.data_params)

path_to_models = os.path.join(config.model_params.model_folder,config.experiment_name, "model_rgbnirh_worldfloodsup_res2vtunet_ep30.pt").replace("\\","/")
model.load_state_dict(load(path_to_models))
model.eval()
model.to("cuda:1")

inference_function = get_model_inference_function(model, config, apply_normalization=True)



def slice_tensor(tensor, size):
    slices = []
    _, height, width = tensor.shape
    x_slices = (width - 1) // size 
    y_slices = (height - 1) // size

    for i in range(x_slices):
        for j in range(y_slices):
            left = i * size
            upper = j * size
            right = min(left + size, width)
            lower = min(upper + size, height)
            slice = tensor[:, upper:lower, left:right]
            # print(slice.shape)
            slices.append(slice)

    return slices

def combine_slices(slices, tensor_size):
    _, height, width = tensor_size    
    x_slices = (width - 1) // slices[0].shape[2] 
    y_slices = (height - 1) // slices[0].shape[1] 
    combined_tensor = torch.zeros(3,height , width)
    
    for i in range(x_slices):
        for j in range(y_slices):
            slice_index = i * y_slices + j
            slice = slices[slice_index]
            left = i * slice.shape[2]
            upper = j * slice.shape[1]
            combined_tensor[:, upper:upper+slice.shape[1], left:left+slice.shape[2]] = slice

    return combined_tensor
    

event_id = "RS2_20161008_Water_Extent_Corail_Pestel.tif"
def inference(event_id, inference_function):
    channel_configuration = config.model_params.hyperparameters.channel_configuration
    dataset_folder = "./worldfloods_v1_0_sample/"
    tiff_s2 = os.path.join(dataset_folder, "val", "S2", event_id)
    tiff_gt = os.path.join(dataset_folder, "val", "gt", event_id)
    tiff_permanentwaterjrc = os.path.join(dataset_folder, "val", "PERMANENTWATERJRC", event_id)
    window = None
    channels = get_channel_configuration_bands(channel_configuration)
    
    # Read inputs
    torch_inputs, transform = dataset.load_input(tiff_s2, window=window, channels=channels)
    # Make predictions
    sliced_images = slice_tensor(torch_inputs, size=256)
    outs = []
    for i in sliced_images:
    outputs = inference_function(i.unsqueeze(0))[0] # (num_classes, h, w)
    outs.append(outputs)
    
    recombined_image = combine_slices(outs, torch_inputs.shape)    
    prediction = torch.argmax(recombined_image, dim=0).long() # (h, w)
    # Mask invalid pixels
    mask_invalid = torch.all(torch_inputs == 0, dim=0)
    prediction+=1
    prediction[mask_invalid] = 0
    
    # Load GT and permanent water for plotting
    torch_targets, _ = dataset.load_input(tiff_gt, window=window, channels=[0])
    torch_permanent_water, _ = dataset.load_input(tiff_permanentwaterjrc, window=window, channels=[0])
    
    sliced_floods = slice_tensor(torch_targets, size=256)
    recombined_floods = combine_slices(sliced_floods, torch_inputs.shape)    
    sliced_water = slice_tensor(torch_permanent_water, size=256)
    recombined_water = combine_slices(sliced_water, torch_inputs.shape)    
    
    return torch_inputs, recombined_floods, recombined_water, transform

    
def plot(torch_inputs, recombined_floods, recombined_water, transform):
    window = None
    fig, axs = plt.subplots(2,2, figsize=(16,16))
    plot_utils.plot_rgb_image(torch_inputs.squeeze(0), transform=transform, ax=axs[0,0])
    axs[0,0].set_title("RGB Composite")
    plot_utils.plot_gt_v1(recombined_floods, transform=transform, title = "Groud truth", ax=axs[0,1])

    plot_utils.plot_gt_v1_with_permanent(recombined_floods, recombined_water, window=window, transform=transform, ax=axs[1,0])
    axs[1,0].set_title("Ground Truth with JRC Permanent")
    plot_utils.plot_gt_v1(prediction.unsqueeze(0),transform=transform, ax=axs[1,1])
    axs[1,1].set_title("Model prediction")
    plt.tight_layout()
