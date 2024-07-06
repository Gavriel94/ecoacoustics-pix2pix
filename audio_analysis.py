import wave
import os
import pandas as pd


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
            print('analysing', file)
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


if not os.path.exists('data/analysis/sm4_data.csv') or \
        not os.path.exists('data/analysis/smmicro_data.csv'):
    sm4_df = pd.DataFrame()
    smmicro_df = pd.DataFrame()
    # SM4 files
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

    # SMMicro files
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI1')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI2')],
                           ignore_index=True)
    smmicro_df = pd.concat([smmicro_df, wav_data('raw_data/2024_03/SMMicro/PLI3')],
                           ignore_index=True)

    sm4_df.to_csv('data/analysis/sm4_data.csv', index=False)
    smmicro_df.to_csv('data/analysis/smmicro_data.csv', index=False)
else:
    sm4_df = pd.read_csv('data/analysis/sm4_data.csv')
    smmicro_df = pd.read_csv('data/analysis/smmicro_data.csv')
# * do more queries here
c = smmicro_df.query('Duration == 60.0')
print(c)
