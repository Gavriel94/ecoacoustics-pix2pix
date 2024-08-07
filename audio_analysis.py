import wave
import os
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s:%(message)s',
                    level=logging.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# save the phase information


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
    all_df = pd.DataFrame()
    for year_dir in os.listdir(data_root):
        for mic_dir in os.listdir(os.path.join(data_root, year_dir)):
            for loc_dir in os.listdir(os.path.join(data_root, year_dir, mic_dir)):
                if loc_dir == 'summaries':
                    continue
                full_path = os.path.join(data_root, year_dir, mic_dir, loc_dir)
                data = wav_data(full_path, verbose=verbose)
                data['mic'] = mic_dir
                data['location'] = loc_dir
                all_df = pd.concat([all_df, data], ignore_index=True)

    save_path = f'{dataset_root}analysis.csv'
    all_df.to_csv(save_path, index=False)
    logging.info(f'Analysis saved in {save_path}')
    return save_path


def get_dict(num_channels: int, sample_rate: int,
             frames: int, bit_depth: int,
             duration: float, wav_path: str,
             file: str, exception: str = None):
    """
    Formats the audio metrics into a dictionary and returns it.
    """
    d = {
        'Path': wav_path,
        'Channels': num_channels,
        'Sample Rate': sample_rate,
        'Frames': frames,
        'Bit Depth': bit_depth,
        'Duration': duration,
        'File': file
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
            logging.info(f'Analysing {file} {i + 1}/{len(files)} files')
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
                             duration, path, file))
        except Exception as e:
            # file is possibly corrupt
            num_channels = w.getnchannels()
            sample_rate = w.getframerate()
            frames = w.getnframes()
            bit_depth = w.getsampwidth() * 8
            dicts.append(get_dict(num_channels, sample_rate,
                         frames, bit_depth,
                         duration, path, file,
                         str(e)))
    return pd.DataFrame(dicts)
