"""
Utility functions

"""
import gzip
import os
import random
import librosa
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import json
import soundfile as sf


def spectrogram_to_audio(spectrogram_path: str, output_path: str, sample_rate):
    # open spectrogram
    spectrogram_img = Image.open(spectrogram_path)
    spectrogram_arr = np.array(spectrogram_img)

    # get magnitude and phase values
    root, specs, img_path = spectrogram_path.split('/')
    params_path = os.path.join(root, specs, 'params', img_path.replace('.png', '.json'))
    # with open(params_path, 'r') as f:
    with gzip.open(params_path, 'wt') as f:  # jsons are zipped
        params = json.load(f)
        magnitude_real = np.array(params['magnitude_real'])
        magnitude_imag = np.array(params['magnitude_imag'])
        phase_real = np.array(params['phase_real'])
        phase_imag = np.array(params['phase_imag'])

        # recreate complex numbers
        magnitude = magnitude_real + 1j * magnitude_imag
        phase = phase_real + 1j * phase_imag

        s_db = (spectrogram_arr / 255) * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)
        s = librosa.db_to_amplitude(s_db, ref=np.max(magnitude))

        # shift sine waves
        stft_matrix = s * phase

        # inverse short term fourier transform
        y = librosa.istft(stft_matrix,
                          hop_length=params['hop_length'],
                          win_length=params['n_fft'],
                          length=params.get('original_length'))

        file_path = os.path.join(output_path, spectrogram_path.split('/')[2].replace('.png', '.wav'))
        # save audio
        sf.write(file_path, y, sample_rate)


def remove_padding(tensor, original_dimensions, pad_coords: dict, is_target):
    try:
        batch_size, _, h, w = tensor.shape
        for i in range(batch_size):
            orig_h, orig_w = original_dimensions[i]
            pad_coords = pad_coords[i]
            if is_target:
                cropped_tensor = tensor[i:i+1, :, pad_coords['top']:pad_coords['top'] + orig_h, :w - pad_coords['right']]
            else:
                cropped_tensor = tensor[i:i+1, :, pad_coords['top']:pad_coords['top'] + orig_h, pad_coords['left']:pad_coords['left'] + orig_w]

            return cropped_tensor
    except ValueError:
        batch_size, h, w = tensor.shape
        for i in range(batch_size):
            orig_h, orig_w = original_dimensions[i]
            pad_coords = pad_coords[i]
            if is_target:
                cropped_tensor = tensor[i:i+1, pad_coords['top']:pad_coords['top'] + orig_h, :w - pad_coords['right']]
            else:
                cropped_tensor = tensor[i:i+1, pad_coords['top']:pad_coords['top'] + orig_h, pad_coords['left']:pad_coords['left'] + orig_w]

            return cropped_tensor


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


def custom_collate(batch):
    input_tensors, target_tensors, original_dimensions, padding_coords, image_path = zip(*batch)
    return torch.stack(input_tensors), torch.stack(target_tensors), original_dimensions, padding_coords, image_path


def save_tensor_as_img(tensor, save_path):
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
    image.save(save_path, format='png')


def save_figure(*data, title, xlabel, ylabel, save_path):
    for d in data:
        plt.plot(d)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)


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


def train_val_test_split(data: list, split_percent: float, shuffle=True):
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
    if len(val) == 0 or len(test) == 0:
        raise ValueError('Not enough data for train/val/test split.\n'
                         'Try a lower split_percent')
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
