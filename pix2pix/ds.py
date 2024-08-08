import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SpectrogramDataset(Dataset):
    def __init__(self, data: list, augment: bool, canvas_size: int):
        self.data = data

        self.augmentation = v2.Compose([
            v2.Resize((224, 224)),
            v2.Pad(padding=4, fill=(0, 0, 0), padding_mode='constant'),
            v2.ToImage(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.normalise_greyscale = v2.Compose([
            v2.ToImage(),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        if (canvas_size & (canvas_size-1) == 0) and canvas_size != 0:
            self.canvas_size = canvas_size
        else:
            raise NotTwoPower('canvas_size must be in the series 2^n ')

    def calculate_bbox(self, image_arr):
        # Find coordinates where pixel value is 255 (white)
        white_pixels = np.where(image_arr == 255)
        if white_pixels[0].size == 0:  # No white area found
            return None

        top = np.min(white_pixels[0])
        bottom = np.max(white_pixels[0])
        left = np.min(white_pixels[1])
        right = np.max(white_pixels[1])

        return top, bottom, left, right

    def crop_canvas(self, image):
        bbox = self.calculate_bbox(image)
        if bbox:
            top, bottom, left, right = bbox
            cropped_img = image.crop((left, top, right + 1, bottom + 1))
        else:
            cropped_img = image  # No white area to crop, return original image

        return cropped_img

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
            if np_arr.ndim == 2:  # Ensure the array has a channel dimension
                np_arr = np.expand_dims(np_arr, axis=-1)  # Add a channel dimension
            np_arr = torch.tensor(np_arr, dtype=torch.float32) / 255.0
            np_arr = np_arr.permute(2, 0, 1)
            np_arr = self.normalise_greyscale(np_arr)
            np_arr = self.to_tensor(np_arr)
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

        padded_img = np.expand_dims(padded_img, axis=-1).astype(np.uint8)

        half_width = self.canvas_size // 2
        # extract left half of the image
        input_arr = padded_img[:, :half_width, :]
        # extract right half of the image
        target_arr = padded_img[:, half_width:, :]

        input_arr = input_arr.squeeze(axis=-1)
        target_arr = target_arr.squeeze(axis=-1)
        Image.fromarray(input_arr).save('tmp/img_left.png')
        Image.fromarray(target_arr).save('tmp/img_right.png')

        if input_arr.shape[1] != target_arr.shape[1]:
            raise Exception(f'Shape mismatch. {input_arr.shape}, {target_arr.shape}')

        input_tensor = convert_permute_transform(input_arr)
        target_tensor = convert_permute_transform(target_arr)
        return input_tensor, target_tensor, (img_height, img_width)


class NotTwoPower(Exception):
    pass
