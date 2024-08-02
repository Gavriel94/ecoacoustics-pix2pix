import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SpectrogramDataset(Dataset):
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
                v2.Normalize(mean=[0.5], std=[0.5]),
                v2.ToDtype(torch.float32, scale=True)
            ])
        self.canvas_size = 512

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

        img_path = self.data[index]
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img)
        img_height, img_width = img_arr.shape

        scale = min(self.canvas_size / img_width, self.canvas_size / img_height)
        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        resized_img = Image.fromarray(img_arr).resize((scaled_width, scaled_height),
                                                      Image.LANCZOS)
        img_arr = np.array(resized_img)

        pad_height = max(0, self.canvas_size - scaled_height)
        pad_width = max(0, self.canvas_size - scaled_width)
        padded_img = np.pad(img_arr, ((0, pad_height), (0, pad_width)),
                            mode='constant', constant_values=255)
        padded_img = np.expand_dims(padded_img, axis=-1)
        half_width = self.canvas_size // 2
        # extract left half of the image
        input_arr = padded_img[:, :half_width, :]
        # extract right half of the image
        target_arr = padded_img[:, half_width:, :]

        if input_arr.shape[1] != target_arr.shape[1]:
            raise Exception(f'Shape mismatch. {input_arr.shape}, {target_arr.shape}')

        input_tensor = convert_permute_transform(input_arr)
        target_tensor = convert_permute_transform(target_arr)
        return input_tensor, target_tensor
