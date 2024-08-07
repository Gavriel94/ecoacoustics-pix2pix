"""
Utility functions

- `get_files()`
- `split_data()`
- `set_device()`
"""
import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def test_custom_l1_loss():
    # copied from train.py
    def custom_l1_loss(input, target, padding_value=1.0):
        if input.size() != target.size():
            target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

        mask = (input != padding_value).float()
        loss = nn.L1Loss(reduction='none')(input, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()

    input_tensor = torch.tensor([
        [[0.5, 1.0, 0.3],
         [1.0, 0.7, 1.0],
         [0.2, 1.0, 0.6]]
    ]).unsqueeze(0)
    target_tensor = torch.tensor([
        [[0.6, 1.0, 0.4],
         [1.0, 0.8, 1.0],
         [0.3, 1.0, 0.5]]
    ]).unsqueeze(0)

    loss = custom_l1_loss(input_tensor, target_tensor)

    non_padding_elements = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.6])
    target_elements = torch.tensor([0.6, 0.4, 0.8, 0.3, 0.5])
    expected_loss = F.l1_loss(non_padding_elements, target_elements, reduction='mean')
    
    print('loss', loss)
    print('expected loss', expected_loss)
    assert torch.isclose(loss, expected_loss, rtol=1e-4)

# def custom_collate(batch):
#     # stops conversion of original_dimensions to tensors
#     input_tensors, target_tensors = zip(*batch)
#     return torch.stack(input_tensors), torch.stack(target_tensors)


def save_tensor(tensor, save_path):
    t = tensor.cpu().detach().numpy()
    try:
        batch_size, channels, height, width = t.shape
        for i in range(batch_size):
            img = np.squeeze(t[i])
    except Exception:
        channels, height, width = t.shape
        img = np.squeeze(t)

    # Normalize to 0-255 range and convert to uint8
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    image = Image.fromarray(img, mode='L')
    image.save(f'{save_path}')


def save_img_arr(img_arr, save_path):
    image = Image.fromarray(img_arr)
    image.save(f'{save_path}')


def plot_loss(disc_loss, gen_loss, l1_loss):
    plt.plot(disc_loss)
    plt.plot(gen_loss)
    plt.plot(l1_loss)
    plt.show()


def get_files(dataset_path: str, include_correlated: bool):
    """
    Returns a list file paths to each image in the dataset.
    """
    files = os.listdir(dataset_path)
    if include_correlated:
        files.extend(os.listdir(os.path.join(dataset_path, 'correlated')))
    files = [file for file in files if file.endswith('.png')]
    files_complete = [os.path.join(dataset_path, file) for file in files]
    return files_complete


def split_data(data: list, split_percent: float, shuffle=True):
    """
    Creates a train, val, test split.

    Uses `split_percent` as the training data, delegates half of
    remaining data to validation and the other half to test.


    Args:
        data (list): List of paths to dataset images
        split_percent (float): _description_
        shuffle (bool, optional): _description_. Defaults to True.

    Returns:
        tuple[list, list, list]: train, val and test image paths.
    """
    if shuffle:
        random.shuffle(data)
    split1 = int(len(data) * split_percent)
    split2 = int((len(data) - split1) / 2)
    train = data[: split1]
    val = data[split1: split1 + split2]
    test = data[split1 + split2: (split1 + (split2 * 2))]
    return train, val, test


def set_device(device: str):
    """
    Set the device to MPS, CUDA or CPU.

    Args:
        mps (bool): Use the GPU, for macOS devices.
        cuda (bool): Use the GPU, for Windows devices.
        cpu (bool): Use the CPU.

    Raises:
        MPSNotAvailable: MPS was chosen but is not available.
        MPSNotBuilt: MPS was chosen and is available but is not built.
        CUDANotAvailable: CUDA was chosen but is not available.
        ValueError: No device was chosen.

    Returns:
        torch.device: Device for model training and inference.
    """
    if device == 'mps':
        mps_avail = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        if not mps_avail:
            raise MPSNotAvailable('MPS is not available.')
        if not mps_built:
            raise MPSNotBuilt('MPS is not built.')
        if mps_avail and mps_built:
            return torch.device('mps')
    elif device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise CUDANotAvailable('CUDA is not available.')
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError('Device must be \'mps\', \'cuda\' or \'cpu\'.')


class MPSNotAvailable(Exception):
    pass


class MPSNotBuilt(Exception):
    pass


class CUDANotAvailable(Exception):
    pass
