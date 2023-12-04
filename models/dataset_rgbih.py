import torch
from torch.utils.data import Dataset, DataLoader

class RGBIH_Dataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        data = {"image": image, "mask": mask}
        return data