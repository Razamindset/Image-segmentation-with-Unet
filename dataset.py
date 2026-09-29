import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
import numpy as np
from PIL import Image
import random


class PetDataset(Dataset):
    def __init__(self, root="./data", split="trainval", transform=None):
        super().__init__()

        self.dataset = datasets.OxfordIIITPet(
            root=root, 
            split=split, 
            target_types="segmentation", 
            download=True
        )

        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, mask = self.dataset[index]

        # Resize both of the images
        img = img.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        # Convert image to float tensor [0, 1]
        img = T.ToTensor()(img)

        # Oxford trimaps contain: 1 (Foreground/Pet), 2 (Background), 3 (Not Classified/Border)
        # Convert to a clean binary mask: Pet (1) -> 1, Background/Border (2, 3) -> 0
        mask_np = np.array(mask)
        binary_mask = (mask_np == 1).astype(np.float32)

        # Convert mask to tensor of shape [1, H, W]
        mask = torch.tensor(binary_mask).unsqueeze(0)

        return img, mask


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, mask = self.base[i]          # img [3,H,W] in 0..1, mask [1,H,W] in {0,1}

        # geometric: SAME transform on image and mask
        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # photometric: image only, never the mask
        img = self.jitter(img)
        return img, mask


if __name__ == "__main__":
    # Initialize dataset (will automatically download if not present)
    train_dataset = PetDataset(root="./data", split="trainval")
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Fetch one batch to verify shapes
    images, masks = next(iter(train_loader))
    
    print(f"Batch Image shape: {images.shape}")  # Expected: torch.Size([4, 3, 256, 256])
    print(f"Batch Mask shape:  {masks.shape}")   # Expected: torch.Size([4, 1, 256, 256])
    print(f"Unique values in mask: {torch.unique(masks)}") # Expected: tensor([0., 1.])