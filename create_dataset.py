"""
Automagically creates a dataset for the Pix2Pix cGAN.

### RAW_DATA_ROOT

The path to the directory containing all of the raw data.
The data must be organised in this way.

Set RAW_DATA_ROOT as path in `config.py`.

```python
raw_data/
└── year/
    ├── microphone_1/
    │   ├── location_1/
    │   └── location_2/
    └── microphone_2/
        ├── location_1/
        └── location_2/
```

### DATASET_ROOT

The generated dataset.

Set DATASET_ROOT as path in `config.py`.

```python
data/
├── analysis/
│   ├── microphone_1_data.csv
│   └── microphone_2_data.csv
├── summary/
│   └── location_data.csv
├── train/
├── val/
├── test/
└── data_dict.json

```

The folder contains

- Analysis Folder
    - audio analysis for matched recordings from both microphones in a queryable format
- Summary Folder
    - Files detailing times and dates in each location where reordings match between microphones
- Train Folder
    - Stitched input/target images composing the training set.
- Validation Folder
    - Stitched input/target images composing the validation set.
- Test Folder
    - Params Folder
        - Dictionaries with magnitude and phase values, for audio recomposition.
    - Stitched input/target images composing the test set.
- Data Dictionary
    - This JSON file shows the view of the data this module uses.
"""

import json
import os
import platform
import random
import gzip

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import correlate2d
from tqdm import tqdm

import config
import utilities as utils
from audio_analysis import analyse_recordings


def create_data_dict(validate_summaries):
    """
    Creates a view of each microphone directory organised by year and microphone.
    If the directory does not contain a summary folder, one is automatically
    created and populated.

    Args:
        validate_summaries (bool): Ensure summary files exist.

    Returns:
        dict: Views of microphone directorys.
    """
    data_dict = {}
    years = [year for year in os.listdir(config.RAW_DATA_ROOT) if year != 'full_summaries']
    for year in years:
        year_path = os.path.join(config.RAW_DATA_ROOT, year)
        mic_dir = os.listdir(year_path)
        for mic in mic_dir:
            year_mic_path = os.path.join(year_path, mic)

            # Validate or create summary files
            if validate_summaries:
                if 'summaries' not in os.listdir(year_mic_path):
                    for loc in os.listdir(year_mic_path):
                        dest = os.path.join(year_mic_path, loc)
                        try:
                            create_summary_file(dest)
                        except UnboundLocalError as e:
                            print(str(e))
                            print(f'Invalid format {dest}')

            fol_dir = os.listdir(year_mic_path)
            for fol in fol_dir:
                full_path = os.path.join(year_mic_path, fol)

                if year not in data_dict:
                    data_dict[year] = {}
                if mic not in data_dict[year]:
                    data_dict[year][mic] = []

                data_dict[year][mic].append(full_path)

    # Save the dictionary as a JSON file
    with open(f'{config.DATASET_ROOT}/data_dict.json', 'w') as f:
        json.dump(data_dict, f)

    return data_dict


def create_summary_file(dir_path):
    """
    Uses date and time extracted from file names to create a summary
    file.

    File names must conform to the format 'mic_YYYYMMDD_HHMMSS', i.e.
    'SM4_20231223_195600' is a SM4 recording taken at 23/12/2023 at
    19:56:00.

    Args:
        dir_path (str): Directory containing recordings.
    """
    def strip_underscore():
        _, date, _ = file.split('_')
        year = date[:4]
        month = date[4:6]
        day = date[6:]
        return f'{year}{month}{day}'

    folder, year, mic, location = dir_path.split('/')
    folder_path = folder + '/' + year + '/' + mic + '/' + 'summaries'
    # create a directory to store the summaries in
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    files = os.listdir(dir_path)
    # sort files by time and date
    files.sort()
    # save dates and times as defined in each wav file name
    datetime_dict = {
        'DATE': [],
        'TIME': []
    }
    first_date = None
    last_date = None
    for i, file in enumerate(files):
        if i == 0:
            first_date = strip_underscore()
        elif i == len(files) - 1:
            last_date = strip_underscore()
        out_date, out_time = utils.filename_to_datetime(file)
        datetime_dict['DATE'].append(out_date)
        datetime_dict['TIME'].append(out_time)
    df = pd.DataFrame(datetime_dict)
    if first_date is None:
        raise ValueError('No start date found.')
    elif last_date is None:
        # it's likely the folder contains only 1 item
        last_date = first_date
    # emulate .txt file name format from other summaries
    file_name = location + '_Summary_' + first_date + '_' + last_date + '.txt'
    file_path = (folder_path + '/' + file_name)
    print(f'Summary file generated\nSaved in {file_path}\n')
    df.to_csv(file_path, index=False)

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
    print(file_name)
    try:
        _, date, time = file_name.split('_')
    except ValueError:
        print(f'Invalid format {file_name}. Likely a summary file.')
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


def match_summaries(summary_paths, verbose=False):
    """
    Inspects pairs of summary files and creates a database of matching datetimes.

    Args:
        summary_paths (list): Paths of summary files.
    """
    def create_save_path():
        """
        Strips the year and location from a full summary path to generate
        an appropriate save directory.
        """
        _, year, _, _, loc = mic1.split('/')
        loc = loc.split('_')[0]
        path = os.path.join(summary_path, f'{year}_{loc}_summary.csv')
        return path

    # sort and collect child paths for each summary directory by location
    loc_sorted = [sorted(os.listdir(file)) for file in summary_paths]
    # pair summary files from corresponding microphones
    paired = list(zip(*loc_sorted))
    # prepend the full path information to the summary file
    paired_paths = [(os.path.join(summary_paths[0], pair[0]),
                     os.path.join(summary_paths[1], pair[1])) for pair in paired]
    summaries = []
    summary_path = os.path.join(config.DATASET_ROOT, 'summary')
    os.makedirs(summary_path, exist_ok=True)
    for pair in paired_paths:
        mic1, mic2 = pair
        matches = match_times(mic1, mic2)
        save_path = create_save_path()
        matches.to_csv(save_path)
        summaries.append(save_path)
        if verbose:
            print(f"Saved matches from {pair[0].split('/')[-1]}"
                  f" and {pair[1].split('/')[-1]} to {save_path}")
    return summaries


def match_times(mic1_summary_path, mic2_summary_path):
    """
    Opens the summary files for both microphones and performs an inner merge to
    find matches.

    Args:
        mic1_summary_path (str): Path to mic1 summary file.
        mic2_summary_path (str): Path to mic2 summary file.

    Returns:
        pd.DataFrame: Columns: DATE and TIME of matching recordings.
    """
    def normalise_df(df):
        """
        Keep date and time columns and ensure there's no duplicates.
        """
        relevant_cols = ['DATE', 'TIME']
        df = df[relevant_cols]
        df = df.drop_duplicates(subset=['TIME'])
        return df

    mic1_df = pd.read_csv(mic1_summary_path)
    mic2_df = pd.read_csv(mic2_summary_path)
    mic1_df = normalise_df(mic1_df)
    mic2_df = normalise_df(mic2_df)
    matching_times = pd.merge(mic1_df, mic2_df, how='inner', on=['DATE',
                                                                 'TIME'])
    return matching_times


def link_recordings(data_dict,
                    summary_files,
                    verbose,
                    file_limit=-1):
    """
    Creates a link between recordings matching from summary files and
    their location in the raw data directory.

    Args:
        data_dict (dict): Data organised by year and microphone.
        summary_files (list): CSV files containing matched recording times.
        verbose (bool): Display progress outputs.
        file_limit (int, optional): Limit recordings from directory.

    Raises:
        ValueError: The folder containing recordings cannot be found.

    Returns:
        set: Full paths to matched recordings from both microphones.
    """
    recording_paths = set()
    for i, summary in enumerate(summary_files):
        _, _, filename = summary.split('/')
        year, month, location, _ = filename.split('_')
        year = '_'.join((year, month))
        # paths to the appropriate recording location for each microphone
        recordings = [path for sublist in list(data_dict[year].values())
                      for path in sublist if location in path]

        # ensure folders containing recordings exist
        if not recordings:
            r_p = os.path.join(config.RAW_DATA_ROOT, year, data_dict.keys()[i], location)
            raise ValueError(f'Cannot find recordings in {r_p}')

        dt_df = pd.read_csv(summary)
        for folder in recordings:
            wav_files = os.listdir(folder)
            wav_files.sort()

            if file_limit != -1:
                # limit number of files
                random.shuffle(wav_files)
                wav_files = wav_files[:file_limit]

            for i, file in enumerate(wav_files):
                date, time = utils.filename_to_datetime(file)
                for _, row in dt_df.iterrows():
                    if row['DATE'] == date and row['TIME'] == time:
                        try:
                            recording_path = os.path.join(folder, file)
                            if verbose:
                                print(f'Linking {recording_path}')
                            recording_paths.add(recording_path)
                        except FileNotFoundError as e:
                            print(str(e))
    return list(recording_paths)


def generate_data(data, n_fft, set_type, correlate, verbose):
    """
    Create a dataset for a Pix2Pix model.

    Generates a spectrogram for each pair of recordings and stitches the input
    on the left and the target on the right.

    There is no border or bleed between the two images.

    The images are saved in `dataset_root/set_type`

    Args:
        data (list(tuple)): mic1 recording at idx[0], mic2 at idx[1].
        n_fft (int): Number of fast Fourier transforms.
        set_type (str): Specify train/val/test.
        verbose (bool): Display progress outputs.
        hop_length (int, optional): Specify hop length. Defaults to -1: hop_length = n_fft // 4.
    """
    # create directory
    dir_path = os.path.join(config.DATASET_ROOT, set_type)
    os.makedirs(dir_path, exist_ok=True)

    save_params = True if set_type == 'test' else False

    for i, (mic1_audio, mic2_audio) in enumerate(tqdm(data, desc="Processing data")):
        mic1_spec = create_spectrogram(mic1_audio,
                                       n_fft,
                                       set_type,
                                       verbose,
                                       save_params=False)  # only need params from the target mic

        mic2_spec = create_spectrogram(mic2_audio,
                                       n_fft,
                                       set_type,
                                       verbose,
                                       save_params=save_params)

        filename = os.path.basename(mic1_audio).replace('.wav', '.png')
        stitch_images(mic1_spec, mic2_spec,
                      save_as=filename, dir_path=dir_path,
                      correlate=correlate)


def stitch_images(mic1_spec, mic2_spec, save_as, dir_path, correlate):
    """
    Stitches mic1 and mic2 spectrograms togethers and saves them.

    Checks run to ensure the dimensions are the same. Cross correlation
    is automatically run and any cross correlated dataset samples are
    saved in their own folder.

    Args:
        mic1_spec (np.array): Spectrogram of mic1.
        mic2_spec (np.array): Spectrogram of mic2
        save_as (str): Save path.
        dir_path (str): Path to parent directory.
    """
    def cross_correlate(spec1: np.array, spec2: np.array):
        """
        Calculates where spec2 best aligns with spec1 by finding the
        x-coordinate with the highest correlation.
        """
        correlation = correlate2d(spec1, spec2, mode='valid')
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
        return x

    separator_width = 0  # not using a separator between images anymore
    separator_colour = (255, 255, 255)
    correlated = False
    if not mic1_spec.shape == mic2_spec.shape:
        if correlate:
            # align them using cross correlation
            offset = cross_correlate(mic1_spec, mic2_spec)
            correlated = True
            # trim spectrograms so they're the same length
            if offset > 0:
                mic1_spec = mic1_spec[:, offset:]
                mic2_spec = mic2_spec[:, :mic1_spec.shape[1]]
            else:
                mic2_spec = mic2_spec[:, -offset:]
                mic1_spec = mic1_spec[:, :mic2_spec.shape[1]]
        else:
            return

    # get dimensions
    mic1_height, mic1_width = mic1_spec.shape
    mic2_height, mic2_width = mic2_spec.shape

    # ensure widths are the same
    min_width = min(mic1_width, mic2_width)
    mic1_spec = mic1_spec[:, :min_width]
    mic2_spec = mic2_spec[:, :min_width]
    if not mic1_spec.shape == mic2_spec.shape:
        raise IndexError('Spectrograms have different dimensions')

    # calculate width of two images
    extended_width = mic1_width * 2 + separator_width
    height = mic1_height

    # stiched aligned spectrograms on the same canvas
    stitched = Image.new('RGB', (extended_width, height), separator_colour)
    mic1_spec_img = Image.fromarray(mic1_spec)  # input image
    mic2_spec_img = Image.fromarray(mic2_spec)  # target image
    stitched.paste(mic1_spec_img, (0, 0))  # stitch input on lhs
    stitched.paste(mic2_spec_img, (mic1_width + separator_width, 0))  # target on rhs

    if correlated:
        # save in a directory for correlated data samples
        correlated_path = os.path.join(dir_path, 'correlated')
        os.makedirs(correlated_path, exist_ok=True)
        output = os.path.join(correlated_path, save_as)
    else:
        output = os.path.join(dir_path, save_as)

    try:
        stitched.save(output)
    except SystemError as se:
        print(f'{str(se)}\nLikely dimensions mismatch\n'
              f'Input dimensions (w x h): {mic1_width, mic1_height}\n'
              f'Target dimensions (w x h):{mic2_width, mic2_height}\n'
              f'Canvas size (w x h): {stitched.width, stitched.height}')
    except Exception as e:
        print(str(e))


def create_spectrogram(wav_file, n_fft, set_type, verbose, save_params):
    """
    Creates a spectrogram from a .wav file.

    Saves magnitude and phase data for spectrograms created for the test
    set to be used for audio recomposition.

    Args:
        wav_file (str): Full path to a .wav file.
        n_fft (int): Number of fast Fourier transformations.
        set_type (str): Specify train/val/test
        verbose (bool): Display progress outputs.
        save_mg (bool): Save data for audio recomposition.

    Returns:
        np.array: Spectrogram as an array.
    """
    hop_length = n_fft // 4
    try:
        y, sr = librosa.load(wav_file)
        s = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

        # normalise spectrogram to range (0, 255)
        s_db_norm = 255 * (s_db - s_db.min()) / (s_db.max() - s_db.min())
        s_db_norm = s_db_norm.astype(np.uint8)

        # data used to recompose back to audio
        if save_params and set_type == 'test':
            params_path = os.path.join(config.DATASET_ROOT, set_type, 'params')
            os.makedirs(params_path, exist_ok=True)

            # STFT parameters and complex spectrum components
            magnitude, phase = librosa.magphase(s)

            os.path.join(config.DATASET_ROOT, 'analysis')

            params = {
                'magnitude_real': magnitude.real.tolist(),
                'magnitude_imag': magnitude.imag.tolist(),
                'phase_real': phase.real.tolist(),
                'phase_imag': phase.imag.tolist(),
                'n_fft': n_fft,
                'hop_length': hop_length,
                'original_length':  len(y),
                'sample_rate': sr,
                'file': wav_file
            }
            params_file_name = wav_file.split('/')[4].replace('.wav', '.json.gz')
            file_path_params = os.path.join(params_path, params_file_name)
            # zip JSONs as they're quite large (170MB+)
            with gzip.open(file_path_params, 'wt', encoding='utf-8') as f:
                json.dump(params, f)
            if verbose:
                print(f'Saving {os.path.basename(file_path_params)}')
    except Exception as e:
        print(str(e))
    return s_db_norm


def pair_recordings(recordings):
    """
    Pairs full paths between matched mic1 and mic2 recordings.

    Args:
        recordings (list): List of recordings.
    """
    def extract_common_part(filepath, mic_symbol):
        filename = filepath.split('/')[-1].split('.')[0]
        # Remove the mic2 symbol
        common_part = filename.replace(mic_symbol, '')
        return common_part

    mic1_spectrograms = []
    mic2_spectrograms = []
    for recording in recordings:
        if not recording.endswith('.wav'):
            continue
        folder, year, mic, loc, filename = recording.split('/')
        if mic == config.INPUT_MIC_NAME:
            mic1_spectrograms.append(recording)
        elif mic == config.TARGET_MIC_NAME:
            mic2_spectrograms.append(recording)

    # Create a dictionary to map the common parts to their respective paths
    dict1 = {extract_common_part(f, config.TARGET_MIC_DELIM): f for f in mic1_spectrograms}
    dict2 = {extract_common_part(f, config.TARGET_MIC_DELIM): f for f in mic2_spectrograms}

    # Find matches and return as a list of tuples
    matched_pairs = [(dict1[key], dict2[key]) for key in dict1 if key in dict2]
    return matched_pairs


def main():
    train_pct = config.TRAIN_PCT
    verbose = True

    # remove hidden files from macOS or Windows systems
    if platform.system() == 'Darwin' or platform.system() == 'Windows':
        utils.remove_hidden_files(config.RAW_DATA_ROOT)

    # create a directory to store the dataset in
    os.makedirs(config.DATASET_ROOT, exist_ok=False)

    print('Analysing recordings')
    # list metrics for each recording and save them in the dataset folder
    analyse_recordings(verbose=verbose)
    print()

    # organise data by year and microphone
    data_dict = create_data_dict(validate_summaries=True)

    # get summary files by date
    summary_paths = {}
    for date, mics in data_dict.items():
        for mic, paths in mics.items():
            for path in paths:
                if 'summaries' in path:
                    if date not in summary_paths:
                        summary_paths[date] = []
                    summary_paths[date].append(path)

    # generate CSV files listing matched recordings from data in summary files
    print('Matching summaries')
    matched_summaries = [match_summaries(summary_paths[year], verbose=verbose)
                         for year in summary_paths.keys()]
    print()

    print('Linking recordings')
    # copy matching recordings from raw data to dataset folders
    recordings = [link_recordings(data_dict, summary_file, verbose=verbose)
                  for summary_file in matched_summaries]
    print()

    # flatten list of lists
    all_recordings = [recording for year in recordings for recording in year]

    paired = pair_recordings(all_recordings)

    train, val, test = utils.train_val_test_split(paired, train_pct, True)

    print('Generating spectrograms for training set')
    generate_data(train, n_fft=4096, set_type='train', correlate=False, verbose=verbose)
    print()

    print('Generating spectrograms for validation set')
    generate_data(val, n_fft=4096, set_type='val', correlate=False, verbose=verbose)
    print()

    print('Generating spectrograms for test set')
    generate_data(test, n_fft=4096, set_type='test', correlate=False, verbose=verbose)
    print()


if __name__ == '__main__':
    main()
