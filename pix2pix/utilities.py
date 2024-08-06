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


def custom_collate(batch):
    # stops conversion of original_dimensions to tensors
    input_tensors, target_tensors, original_dimensions = zip(*batch)
    return torch.stack(input_tensors), torch.stack(target_tensors), original_dimensions


def save_img_tensor_in_tmp(img_tensor, img_type):
    img_arr = img_tensor.cpu().detach().numpy()
    print('img_arr.shape', img_arr.shape)
    try:
        batch_size, channels, height, width = img_arr.shape
    except Exception:
        print(img_arr.shape)
        pass
    print('batch_size', batch_size)
    print('channels', channels)
    print('height', height)
    print('width', width)

    for i in range(batch_size):
        img = np.squeeze(img_arr[i])
    
    # Normalize to 0-255 range and convert to uint8
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    image = Image.fromarray(img, mode='L')
    image.save(f'tmp/input_output/{img_type}_batch_{i}.png')


def save_img_arr_in_tmp(img_arr, img_path, mic_name):
    image = Image.fromarray(img_arr)
    save_path = img_path.split('/')[2]
    image.save(f'tmp/{mic_name}-{save_path}.png')


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
