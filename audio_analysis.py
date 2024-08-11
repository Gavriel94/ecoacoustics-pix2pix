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

import wave
import os
import pandas as pd


def analyse_recordings(data_root: str, dataset_root: str, verbose: bool = False):
    """
    Record the number of channels, sample rate, number of frames, sample
    width and duration for all audio files.
    Aggregates the data from the same microphones to keep the number of
    files to a minimum.

    Args:
        data_root (str): Path to raw data.
        dataset_root (str): Path to the dataset.
        verbose (bool, optional): Output the save path.
    """
    os.makedirs(f'{dataset_root}analysis', exist_ok=True)
    for year_dir in os.listdir(data_root):
        for mic_dir in os.listdir(os.path.join(data_root, year_dir)):
            df = pd.DataFrame()
            for loc_dir in os.listdir(os.path.join(data_root, year_dir, mic_dir)):
                if loc_dir == 'summaries':
                    continue
                full_path = os.path.join(data_root, year_dir, mic_dir, loc_dir)
                df = pd.concat([df, wav_data(full_path, verbose=verbose)], ignore_index=True)
                save_path = f'{dataset_root}analysis/{mic_dir}_data.csv'
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
            # file is possibly corrupt, save exception
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
