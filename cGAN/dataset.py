import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CGANDataset(Dataset):
    def __init__(self, data: list, augment: bool):
        self.data = data
        if augment:
            self.transforms = v2.Compose([
                v2.Resize((224, 224)),
                v2.Pad(padding=4, fill=(0, 0, 0), padding_mode='constant'),
                v2.ToImage(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                v2.ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                v2.ToDtype(torch.float32, scale=True)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Gets the image from the dataset and applies transformations
        before returning it as a tensor.

        Args:
            index (int): Index of dataset.

        Returns:
            Image: _description_
        """
        def convert_permute_transform(np_arr):
            np_arr = torch.tensor(np_arr, dtype=torch.float32) / 255.0
            np_arr = np_arr.permute(2, 0, 1)
            np_arr = self.transforms(np_arr)
            return np_arr

        image_path = self.data[index]
        image = Image.open(image_path).convert('L')
        image_width, _ = image.size
        image_width = image_width // 2
        image_arr = np.array(image)
        image_arr = np.expand_dims(image_arr, axis=-1)
        # extract left half of the image
        input_arr = image_arr[:, :image_width, :]
        # extract right half of the image
        target_arr = image_arr[:, image_width:, :]
        return convert_permute_transform(input_arr), convert_permute_transform(target_arr)
