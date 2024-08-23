"""
# Utility functions

## File Management

- remove_hidden_files - Delete metadata files that Windows or macOS generate.
- filename_to_datetime - Convert filenames. PLI1_20240316_122600 -> ('2024-Mar-16', '12:26:00')
- train_val_test_split - Creates a train, val, test split.
- get_raw_audio - Get full paths to raw audio files from generated audio file names.
"""
import os
import random

import config


def remove_hidden_files(data: str):
    """
    Removes directory metadata files that Windows or macOS generate.
    Iterates through all child directories.

    Args:
        data (str): Path to folder.
    """
    remove_files = ['.DS_Store', 'desktop.ini', 'Thumbs.db']
    for root, dirs, files in os.walk(data):
        for file in files:
            if file in remove_files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f'Error removing {file_path}: {e}.\n'
                          'Try removing manually.')


def filename_to_datetime(file_name: str):
    """
    Translates datetime from filenames into a string format.

    A file_name 2023_1128_145050 would lead to the date being
    represented as 2023-Nov-28 and the time as 14:50:50.

    Args:
        file_name (str): The recordings file name.

    Returns:
        tuple(str, str): Date and time of the recording.
    """
    months = {
        '01': 'Jan',
        '02': 'Feb',
        '03': 'Mar',
        '04': 'Apr',
        '05': 'May',
        '06': 'Jun',
        '07': 'Jul',
        '08': 'Aug',
        '09': 'Sep',
        '10': 'Oct',
        '11': 'Nov',
        '12': 'Dec'
    }
    _, date, time = file_name.split('_')
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    hours = time[:2]
    mins = time[2:4]
    secs = time[4:6]
    # emulate date format from other summaries
    date = year + '-' + months[month] + '-' + day
    # emulate time format from other summaries
    time = hours + ':' + mins + ':' + secs
    return date, time


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


def get_raw_audio(audio_filename: str):
    """
    Find the full path to a raw audio file using an audio filename.

    Args:
        audio_filename (str): basename of an audio file.
    """
    def get_audio(sample):
        # key which is just file basename without ext
        key = sample.replace('.png', '')
        # key where basename matches original target mic format
        audio_key = key.split('_')
        audio_key[0] = audio_key[0] + config.TARGET_MIC_DELIM
        audio_key = '_'.join(audio_key)
        # use data from filename to navigate raw data folder
        print(sample)
        loc, date, time = sample.split('_')
        year = date[:4] + '_' + date[4:6]
        return os.path.join(config.RAW_DATA_ROOT, year, config.TARGET_MIC_NAME, loc, audio_key)

    if isinstance(audio_filename, list):
        audio_paths = []
        for sample in audio_filename:
            audio_paths.append(get_audio(sample))
        return audio_paths
    else:
        return get_audio(audio_filename)
