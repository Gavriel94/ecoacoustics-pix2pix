"""
# Model Utilities

## Transformations

- spectrogram_to_audio - Recompose audio using magnitude and phase data.
- remove_padding_from_tensor - Remove the padding value from a tensor.
- convert_tensor_to_img - Convert a tensor to a PIL image. Values are normalised to range 0-255.
- save_tensor_as_img - `convert_tensor_to_img` then save.
- create_line_graph - Create a simple line graph. Multiple lines can be on one figure.

## Helper Functions

- set_device - Set the device to MPS, CUDA or CPU.
- train_collate - Set how data is passed from the DataLoader to the model during training.
- eval_collate - Set how data is passed from the DataLoader to the model during evaluation.
"""

import gzip
import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from PIL import Image


def spectrogram_to_audio(spectrogram_img: str, params_path: str, file_path: str):
    """
    Recompose audio using magnitude and phase data retained during
    spectrogram creation.

    Args:
        spectrogram_img (str): Spectrogram as PIL image.
        params_path (str): Path to zipped JSON parameter file.
        file_path (str): Where to save the audio.
        sample_rate (int): Sample rate in Hz.
    """
    # open spectrogram
    spectrogram_arr = np.array(spectrogram_img)

    # with open(params_path, 'r') as f:
    with gzip.open(params_path, 'rt', encoding='utf-8') as f:  # jsons are zipped
        params = json.load(f)
        magnitude_real = np.array(params['magnitude_real'])
        magnitude_imag = np.array(params['magnitude_imag'])
        phase_real = np.array(params['phase_real'])
        phase_imag = np.array(params['phase_imag'])

        # recreate complex numbers
        magnitude = magnitude_real + 1j * magnitude_imag
        phase = phase_real + 1j * phase_imag

        s_db = ((spectrogram_arr / 255)
                * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude))
        s = librosa.db_to_amplitude(s_db, ref=np.max(magnitude))

        # shift sine waves
        stft_matrix = s * phase

        # inverse short term fourier transform
        y = librosa.istft(stft_matrix,
                          hop_length=params['hop_length'],
                          win_length=params['n_fft'],
                          length=params['original_length'])

        # save audio
        sf.write(file_path.replace('.png', '.wav'), y, params['sample_rate'])


def remove_padding_from_tensor(tensor, original_dimensions, pad_coords: dict, is_target):
    """
    Remove the padding value from a tensor.

    Args:
        tensor (torch.tensor): Image tensor.
        original_dimensions (tuple): Original width and height of the image.
        pad_coords (dict): Coordinates where the padding stretched to.
        is_target (bool): Target's on the right side of the image so sliced differently.

    Returns:
        torch.tensor: Tensor with the padding value removed from specified coordinates.
    """
    try:
        batch_size, _, h, w = tensor.shape
        for i in range(batch_size):
            # crop tensors with channel values
            orig_h, orig_w = original_dimensions[i]
            pad_coords = pad_coords[i]
            if is_target:
                cropped_tensor = tensor[i:i+1, :, pad_coords['top']:pad_coords['top']
                                        + orig_h, :w - pad_coords['right']]
            else:
                cropped_tensor = tensor[i:i+1, :, pad_coords['top']:pad_coords['top']
                                        + orig_h, pad_coords['left']:pad_coords['left'] + orig_w]
            return cropped_tensor
    except ValueError:
        # crop tensors without channel values
        batch_size, h, w = tensor.shape
        for i in range(batch_size):
            orig_h, orig_w = original_dimensions[i]
            pad_coords = pad_coords[i]
            if is_target:
                cropped_tensor = tensor[i:i+1, pad_coords['top']:pad_coords['top']
                                        + orig_h, :w - pad_coords['right']]
            else:
                cropped_tensor = tensor[i:i+1, pad_coords['top']:pad_coords['top']
                                        + orig_h, pad_coords['left']:pad_coords['left'] + orig_w]
            return cropped_tensor


def convert_tensor_to_img(tensor):
    """
    Normalise pixel values to 0-255, convert to a PIL image and return.
    """
    t = tensor.cpu().detach().numpy()  # convert tensor to np array
    # remove dimension values of 1
    try:
        batch_size, channels, height, width = t.shape
        for i in range(batch_size):
            img = np.squeeze(t[i])
    except Exception:
        channels, height, width = t.shape
        img = np.squeeze(t)

    # Normalize to 0-255 range and convert to uint8
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    # convert to PIL image and save
    image = Image.fromarray(img, mode='L')
    return image


def save_tensor_as_img(tensor, save_path):
    """
    Normalise pixel values to 0-255, convert to a PIL image and save.

    Args:
        tensor (torch.Tensor): Image.
        save_path (str): Where to save the image. '.png' is not required.
    """
    image = convert_tensor_to_img(tensor)
    image.save(save_path, format='png')


def create_line_graph(*data, title, xlabel, ylabel, save_path, legend=None):
    """
    Saves data on a matplotlib line graph. Accepts a variable number of data
    to plot enabling lines on the same graph.

    Args:
        title (str): Figure title.
        xlabel (str): Title for x axis.
        ylabel (str): Title for y axis.
        save_path (str): Path to save figure.
    """
    for d in data:
        plt.plot(d)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    plt.savefig(save_path)
    plt.close()


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


def train_collate(batch):
    """
    Defines how data is passed from the Dataset to the DataLoader.

    Args:
        batch (list): Batch of data.

    Returns:
        tuple: Data to be used in the training loop.
    """
    input_tensors, target_tensors, original_dimensions, padding_coords, image_path = zip(*batch)
    return (torch.stack(input_tensors), torch.stack(target_tensors),
            original_dimensions, padding_coords, image_path)


def eval_collate(batch):
    (input_tensors, target_tensors, original_dimensions,
     padding_coords, image_path, params_path, audio_path) = zip(*batch)
    return (torch.stack(input_tensors), torch.stack(target_tensors),
            original_dimensions, padding_coords, image_path, params_path, audio_path)


class MPSNotAvailable(Exception):
    pass


class MPSNotBuilt(Exception):
    pass


class CUDANotAvailable(Exception):
    pass
