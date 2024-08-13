"""
Model utility functions.

- set_device
- get_files
- train_val_test_split
- custom_collate
- get_test_sample
- spectrogram_to_audio
- remove_padding
- save_tensor_as_img
- save_figure
"""
import gzip
import json
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from PIL import Image


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


def get_files(dataset_path: str, include_correlated: bool):
    """
    Gets a list of file paths to each image in the dataset.

    Args:
        dataset_path (str): Path to dataset root.
        include_correlated (bool): Include images that have been cross correlated.

    Returns:
        list: List of full paths to image data.
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
        split_percent (float): Percent of data used for training
        shuffle (bool, optional): Shuffle before splitting. Defaults to True.

    Returns:
        tuple[list, list, list]: train, val and test image paths.
    """
    if shuffle:
        random.shuffle(data)

    total_count = len(data)
    train_count = int(total_count * split_percent)
    val_count = (total_count - train_count) // 2

    train = data[:train_count]
    val = data[train_count:train_count + val_count]
    test = data[train_count + val_count:]

    if len(val) == 0 or len(test) == 0:
        raise ValueError('Not enough data for train/val/test split.\n'
                         'Try a lower split_percent')
    return train, val, test


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


def get_raw_audio(generated_audio, raw_data_root, mic2_name, mic2_delim):
    """
    Get full paths to raw audio files from generated file names.

    Args:
        generated_audio_dir (_type_): _description_
    """
    audio_paths = []
    for i, sample in enumerate(generated_audio):
        # key which is just file basename without ext
        key = sample.replace('.png', '')
        # key where basename matches original target mic format
        audio_key = key.split('_')
        audio_key[0] = audio_key[0] + mic2_delim
        audio_key = '_'.join(audio_key)
        # use data from filename to navigate raw data folder
        loc, date, time = sample.split('_')
        year = date[:4] + '_' + date[4:6]
        # save path to audio
        audio_paths.append(os.path.join(raw_data_root, year, mic2_name, loc, audio_key))
    return audio_paths


def get_test_batch(dataset_root, raw_data_root, mic2_name, mic2_delim, num_samples):
    """
    Get random samples from the test set and return their full image path in the
    raw data folder and its magnitude and phase parameter dictionary.
    """
    test_dir = os.path.join(dataset_root, 'test')
    samples = [file for file in os.listdir(test_dir)
               if file.endswith('.png')]
    if len(samples) == 0:
        raise Exception('Empty test folder')
    if len(samples) < num_samples:
        print(f'{num_samples} samples requested but {len(samples)} in folder. Returning all.')
        num_samples = len(samples)

    random_samples = random.sample(samples, k=num_samples)

    batch_samples = []
    for i, sample in enumerate(random_samples):
        # key which is just file basename without ext
        key = sample.replace('.png', '')
        # key where basename matches original target mic format
        audio_key = key.split('_')
        audio_key[0] = audio_key[0] + mic2_delim
        audio_key = '_'.join(audio_key)
        # use data from filename to navigate raw data folder
        loc, date, time = sample.split('_')
        year = date[:4] + '_' + date[4:6]

        # save path to spectrogram
        spectrogram_path = os.path.join(test_dir, sample)
        # save path to audio
        audio_path = os.path.join(raw_data_root, year, mic2_name, loc, audio_key + '.wav')
        # save path to params
        params_dict = os.path.join(dataset_root, 'test', 'params', audio_key + '.json.gz')

        # ensure files exist
        if not os.path.exists(spectrogram_path):
            raise FileNotFoundError(f"Couldn't find {spectrogram_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Couldn't find {audio_path}")
        if not os.path.exists(params_dict):
            raise FileNotFoundError(f"Couldn't find {params_dict}")

        batch_samples.append((spectrogram_path, params_dict, audio_path))
        print(f'Processed {i + 1}/{len(random_samples)} samples.')
    return batch_samples


def compare_audio(generated_audio, raw_audio):
    # get metrics for raw audio i.e. bird net and other stuff
    # get metrics for generated audio, i.e. bird net and stuff
    # quantify the difference
    pass


def spectrogram_to_audio(spectrogram_img: str, params_path: str, file_path: str, sample_rate):
    """
    Recompose audio using magnitude and phase data retained during
    spectrogram creation.

    Args:
        spectrogram_img (str): Spectrogram as PIL image.
        output_path (str): Where to save the audio.
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
                          length=params.get('original_length'))

        # save audio
        sf.write(file_path.replace('.png', '.wav'), y, sample_rate)


def remove_padding(tensor, original_dimensions, pad_coords: dict, is_target):
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
    image.save(save_path, format='png')


def save_figure(*data, title, xlabel, ylabel, save_path):
    """
    Saves data on a matplotlib line graph. Accepts a variable number of data
    to plot enabling lines on the same graph.

    Args:
        title (str): Figure title.
        xlabel (list): Title for x axis.
        ylabel (list): Title for y axis.
        save_path (str): Path to save figure.
    """
    for d in data:
        plt.plot(d)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)


class MPSNotAvailable(Exception):
    pass


class MPSNotBuilt(Exception):
    pass


class CUDANotAvailable(Exception):
    pass
