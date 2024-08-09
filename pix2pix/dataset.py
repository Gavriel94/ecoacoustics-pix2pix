import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import math
import os


class Pix2PixDataset(Dataset):
    def __init__(self, dataset: list, use_correlated: bool, augment: bool):
        
        # data is a list of paths like data/train/training_spectrogram.png
        self.data = [os.path.join(dataset, file) for file in os.listdir(dataset) if file.endswith('.png')]
        if use_correlated:
            if 'correlated' in os.listdir(dataset):
                if len(os.listdir(os.path.join(dataset, 'correlated'))) == 0:
                    raise Exception(f"Empty folder {os.path.join(dataset, 'correlated')}")
                for file in os.listdir(os.path.join(dataset, 'correlated')):
                    self.data.append(os.path.join(dataset, 'correlated', file))
            else:
                raise FileNotFoundError('No correlated directory.')
        
        self.augmentation = v2.Compose([
            v2.Resize((224, 224)),
            v2.Pad(padding=4, fill=(0, 0, 0), padding_mode='constant'),
            v2.ToImage(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.normalise = v2.Compose([
            v2.ToImage(),
            # v2.Normalize(mean=[0.5], std=[0.5])
        ])
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def calculate_padding_dimensions(self, img_shape):
        def next_power_of_2(x):
            return 2 ** math.ceil(math.log2(x))
        
        width, height = img_shape
        target_width = max(next_power_of_2(width), width)
        target_height = max(next_power_of_2(height), height)

        return target_width, target_height

    def pad_image(self, image_arr, target_width, target_height):
        pad_width = target_width - image_arr.shape[1]
        pad_height = target_height - image_arr.shape[0]

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        padding_coords = {
            'left': pad_left,
            'right': pad_right,
            'top': pad_top,
            'bottom': pad_bottom
        }

        return np.pad(image_arr,
                      ((pad_top, pad_bottom),
                       (pad_left, pad_right)),
                      mode='constant',
                      constant_values=255), padding_coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        
        print('Retrieving image', image_path)
        image = Image.open(image_path).convert('L')
        image_arr = np.array(image)

        target_width, target_height = self.calculate_padding_dimensions(image_arr.shape)

        padded_image, padding_coords = self.pad_image(image_arr,
                                                      target_width,
                                                      target_height)

        padded_input = padded_image[:, :padded_image.shape[1] // 2]
        padded_target = padded_image[:, padded_image.shape[1] // 2:]

        input_tensor = self.to_tensor(padded_input)
        target_tensor = self.to_tensor(padded_target)
        original_size = image_arr.shape
        
        _, _, image_file = image_path.split('/')

        return input_tensor, target_tensor, original_size, padding_coords, image_file


class NotTwoPower(Exception):
    pass
