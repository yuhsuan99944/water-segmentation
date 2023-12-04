from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        return image

def collate_fn(batch):
    images = torch.stack(batch, dim=0)
    return {'image': images}

data = torch.randn(10, 3, 224, 224)
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

for batch in dataloader:
    images = batch['image']
    print(images.shape)
