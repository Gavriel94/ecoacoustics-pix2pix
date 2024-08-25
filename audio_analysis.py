"""
Create a database containing basic information about each recording.
- Full path
- Number of channels
- Sample rate
- Number of frames
- Bit depth
- Recording duration

These metrics help accurate recomposition of audio from spectrograms
and promote a better understanding of the data before any engineering.
"""

import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io.wavfile import read

import config


def analyse_recordings(verbose: bool = False):
    """
    Record the number of channels, sample rate, number of frames, sample
    width and duration for all audio files.

    A file is generated for each microphone.

    Args:
        verbose (bool, optional): Output the save path.
    """
    os.makedirs(os.path.join(config.DATASET_ROOT, 'analysis'), exist_ok=True)
    # ignore summary files
    years = [f for f in os.listdir(config.RAW_DATA_ROOT) if f != 'full_summaries']
    for year_dir in years:
        for mic_dir in os.listdir(os.path.join(config.RAW_DATA_ROOT, year_dir)):
            df = pd.DataFrame()
            for loc_dir in os.listdir(os.path.join(config.RAW_DATA_ROOT, year_dir, mic_dir)):
                if loc_dir == 'summaries':
                    continue
                full_path = os.path.join(config.RAW_DATA_ROOT, year_dir, mic_dir, loc_dir)
                df = pd.concat([df, wav_data(full_path, verbose=verbose)], ignore_index=True)
                save_path = os.path.join(config.DATASET_ROOT, 'analysis', f'{mic_dir}.csv')
                df.to_csv(save_path, index=False)
                print(f'Analysis added to {save_path}')


def get_dict(num_channels: int, sample_rate: int,
             frames: int, bit_depth: int,
             duration: float, wav_path: str,
             exception: str = None):
    """
    Formats the audio metrics into a dictionary and returns it.
    """
    d = {
        'Path': wav_path,
        'Channels': num_channels,
        'Sample Rate': sample_rate,
        'Frames': frames,
        'Bit Depth': bit_depth,
        'Duration': duration
    }
    if exception is not None:
        d['Exception'] = exception
    return d


def wav_data(directory_path: str, verbose: bool = False):
    """
    Record the number of channels, sample rate, number of frames, sample
    width and duration of each audio file in a directory.

    Args:
        directory_path (str): Path to the recordings.
        verbose (bool, optional): . Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing metrics.
    """
    files = os.listdir(directory_path)
    dicts = []
    for i, file in enumerate(files):
        if verbose:
            print(f'Analysing {file} {i + 1}/{len(files)} files')
        path = os.path.join(directory_path, file)
        try:
            with wave.open(path, 'rb') as w:
                num_channels = w.getnchannels()
                sample_rate = w.getframerate()
                frames = w.getnframes()
                bit_depth = w.getsampwidth() * 8  # convert bytes to bits

                duration = frames / float(sample_rate)
                dicts.append(get_dict(num_channels, sample_rate,
                             frames, bit_depth,
                             duration, path))
        except Exception as e:
            # file is possibly corrupt, check error
            exception = str(e)
            num_channels = w.getnchannels()
            sample_rate = w.getframerate()
            frames = w.getnframes()
            bit_depth = w.getsampwidth() * 8
            dicts.append(get_dict(num_channels, sample_rate,
                         frames, bit_depth,
                         duration, path,
                         exception))
    return pd.DataFrame(dicts)


def plot_wav(file_path):
    # Read the WAV file
    sample_rate, data = read(file_path)

    # Create a time array
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Plot the audio data
    plt.figure(figsize=(12, 6))
    plt.plot(time, data)
    plt.title('SMMicro PLI2 28/11/2023 - 12:50:00')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.savefig('e.png')


plot_wav('raw_data/2023_11/SMMicro/PLI2/PLI2_20231128_125000.wav')
