import wave
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def gather_files():
    sm4_df = pd.DataFrame()
    smmicro_df = pd.DataFrame()

    sm4_df = pd.concat([sm4_df, wav_data('raw_data/2023_11/SM4/PLI2')],
                       ignore_index=True)
    sm4_df = pd.concat([sm4_df, wav_data('raw_data/2023_11/SM4/PLI3')],
                       ignore_index=True)
    sm4_df = pd.concat([sm4_df, wav_data('raw_data/2024_03/SM4/PLI1')],
                       ignore_index=True)
    sm4_df = pd.concat([sm4_df, wav_data('raw_data/2024_03/SM4/PLI2')],
                       ignore_index=True)
    sm4_df = pd.concat([sm4_df, wav_data('raw_data/2024_03/SM4/PLI3')],
                       ignore_index=True)

    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2023_11/SMMicro/PLI2')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2023_11/SMMicro/PLI3')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI1')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI2')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI3')],
                           ignore_index=True)

    return sm4_df, smmicro_df


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
    Analyse each wav file from a directory and return a pd.DataFrame
    containing metrics from each file.
    """
    files = os.listdir(directory_path)
    dicts = []
    for file in files:
        if verbose:
            print('Analysing', file)
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
            # file is possibly corrupt
            num_channels = w.getnchannels()
            sample_rate = w.getframerate()
            frames = w.getnframes()
            bit_depth = w.getsampwidth() * 8
            dicts.append(get_dict(num_channels, sample_rate,
                         frames, bit_depth,
                         duration, path,
                         str(e)))
    return pd.DataFrame(dicts)


def create_spectrograms(directory_path: str,
                        n_fft: int,
                        hop_length: int = None,
                        verbose: bool = False):
    """
    Create spectrograms for all .wav files in a folder.

    Args:
        directory_path (str): Path to folder containing .wav files.
        n_fft (int): Number of fast Fourier transform points.
        hop_length (int, optional): Number of samples in each frame.
            Default is n_fft // 4.
        verbose (bool, optional): Display progress. Defaults to False.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    files = os.listdir(directory_path)
    for i, file in enumerate(files):
        if verbose:
            print(f'Analysing {file}, {i + 1}/{len(files)}')
        path = os.path.join(directory_path, file)
        try:
            y, sr = librosa.load(path)
            s = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(s_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='log', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{file.replace('.wav', '')} Spectrogram")
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            save_path = directory_path.replace('data/', 'data/analysis/spectrograms/')
            plt.savefig(f"{save_path}/{file.replace('.wav', '.png')}")
            plt.close()
        except Exception as e:
            print(str(e))


def run_queries(sm4_df: pd.DataFrame,
                smmicro_df: pd.DataFrame):
    pass
    # c = smmicro_df.query('Duration == 60.0')
    # print(c)


if not os.path.exists('data/analysis/sm4_data.csv') \
        or not os.path.exists('data/analysis/smmicro_data.csv'):
    sm4_df, smmicro_df = gather_files()
    sm4_df.to_csv('data/analysis/sm4_data.csv', index=False)
    smmicro_df.to_csv('data/analysis/smmicro_data.csv', index=False)
else:
    sm4_df = pd.read_csv('data/analysis/sm4_data.csv')
    smmicro_df = pd.read_csv('data/analysis/smmicro_data.csv')

    run_queries(sm4_df, smmicro_df)

    create_spectrograms('data/2023_11/SM4/PLI2', n_fft=4096, verbose=True)
    create_spectrograms('data/2023_11/SM4/PLI3', n_fft=4096, verbose=True)
    create_spectrograms('data/2023_11/SMMicro/PLI2', n_fft=4096, verbose=True)
    create_spectrograms('data/2023_11/SMMicro/PLI3', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SM4/PLI1', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SM4/PLI2', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SM4/PLI3', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SMMicro/PLI1', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SMMicro/PLI2', n_fft=4096, verbose=True)
    create_spectrograms('data/2024_03/SMMicro/PLI3', n_fft=4096, verbose=True)
