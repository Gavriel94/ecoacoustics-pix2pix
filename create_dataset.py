import pandas as pd
import os
import librosa
import numpy as np
import platform
import shutil
import json
from PIL import Image
from scipy.signal import correlate2d
import random

# from pix2pix.config import setup_logging
from audio_analysis import analyse_recordings
from pix2pix.utilities import train_val_test_split
import logging

# setup_logging()

logging.basicConfig(format='%(asctime)s:%(message)s',
                    level=print,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger('PIL').setLevel(logging.WARNING)


def remove_hidden_files(data_path: str):
    """
    Removes hidden files that Windows or macOS may generate.
    """
    file_names = ['.DS_Store', 'desktop.ini', 'Thumbs.db']
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file in file_names:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f'Error removing {file_path}: {e}.\n'
                          'Try removing manually.')


def create_data_dict(data_from: str, data_to: str, save_json=True):
    """
    Creates a dictionary of paths to recordings organised by year and microphone.
    Optionally saves the dictionary as a JSON file.

    Args:
        data_from (str): Path of source data.
        data_to (str): Path where dataset will be stored.
        save_json (bool, optional): Saves data as JSON. Defaults to True.

    Returns:
        dict: Data organised by year and microphone.
    """
    data_dict = {}
    years = os.listdir(data_from)
    for year in years:
        year_path = os.path.join(data_from, year)
        mic_dir = os.listdir(year_path)
        for mic in mic_dir:
            year_mic_path = os.path.join(year_path, mic)
            # Check for a summary folder and create one if it does not exist
            validate_summary_folder(year_mic_path)
            fol_dir = os.listdir(year_mic_path)
            for fol in fol_dir:
                full_path = os.path.join(year_mic_path, fol)

                if year not in data_dict:
                    data_dict[year] = {}
                if mic not in data_dict[year]:
                    data_dict[year][mic] = []

                data_dict[year][mic].append(full_path)
    if save_json:
        with open(f'{data_to}/data_dict.json', 'w') as f:
            json.dump(data_dict, f)
    return data_dict


def get_paths(data: dict, year: str, mic: str = None):
    """
    Get paths to all information by year or microphone.

    Args:
        data_dict (dict): Data organised by year and microphone.
        year (str): The paths corresponding to a specific year.
        mic (str, optional): Paths corresponding to a specific microphone.
        Defaults to None.

    Returns:
        list | dict: List of paths if microphone is specified otherwise
        a dictionary with the microphones and their respective data paths.
    """
    if mic:
        return data.get(year, {}).get(mic, [])
    return data.get(year, {})


def validate_summary_folder(year_mic_path: str):
    """
    Ensures each micrphone directory contains a populated summary folder.
    If some summary files are missing it may be easier to delete the entire
    summaries folder and create a new data dictionary.
    This will  create a new folder with a summary file for each location.

    Args:
        year_mic_path (str): Full path from root to the microphone.

    Returns:
        str: Path to the summary folder.
    """
    if 'summaries' not in os.listdir(year_mic_path):
        for loc in os.listdir(year_mic_path):
            dest = os.path.join(year_mic_path, loc)
            print(f'Creating summary for {dest}')
            create_summary_file(dest)
    return os.path.join(year_mic_path, 'summaries')


def create_summary_file(dir_path: str):
    """
    Extracts date and time from each .wav file name to create a new summary
    file. File names must conform to the format 'mic_YYYYMMDD_HHMMSS'.
    For example 'SM4_20231223_195600' is a SM4 recording taken at 23/12/2023 at
    19:56:00.

    Args:
        dir_path (str): Directory containing .wav files.
    """
    def format_date(wav_file_name):
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
    dic = {
        'DATE': [],
        'TIME': []
    }
    first_date = None
    last_date = None
    for i, file in enumerate(files):
        if i == 0:
            first_date = format_date(file)
        elif i == len(files) - 1:
            last_date = format_date(file)
        out_date, out_time = format_file_name(file)
        dic['DATE'].append(out_date)
        dic['TIME'].append(out_time)
    df = pd.DataFrame(dic)
    if first_date is None:
        raise ValueError('No start date found.')
    elif last_date is None:
        # folder likely contains 1 item
        last_date = first_date
    # emulate .txt file name format from other summaries
    file_name = location + '_Summary_' + first_date + '_' + last_date + '.txt'
    file_path = (folder_path + '/' + file_name)
    print(f'Summary file generated\nSaved in {file_path}\n')
    df.to_csv(file_path, index=False)


def format_file_name(file_name: str):
    """
    Extracts the date and time from a recordings file name.

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


def match_summaries(summary_dir: list, data_to: str, verbose: bool = False):
    """
    Finds matching times from different microphones for each location-based
    summary file.

    Args:
        summary_dir (list): List of summary directories for a year.
        data_to (str): Directory where dataset will be stored.
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
    loc_sorted = [sorted(os.listdir(file)) for file in summary_dir]
    # pair summary files from corresponding microphones
    paired = list(zip(*loc_sorted))
    # prepend the full path information to the summary file
    paired_paths = [(os.path.join(summary_dir[0], pair[0]),
                     os.path.join(summary_dir[1], pair[1])) for pair in paired]
    summaries = []
    summary_path = os.path.join(data_to, 'summary')
    os.makedirs(summary_path, exist_ok=True)
    for pair in paired_paths:
        mic1, mic2 = pair
        matches = match_times(mic1, mic2)
        save_path = create_save_path()
        matches.to_csv(save_path)
        summaries.append(save_path)
        if verbose:
            print(f'Saved matches from {pair} to {save_path}')
    return summaries


def match_times(mic1_summary_path: str, mic2_summary_path: str):
    """
    Inspects summary files to find recordings taken at the same time and date.

    Args:
        sm4_summary_path (str): Path to SM4 summary file.
        smmicro_summary_path (str): Path to SMMicro summary file.

    Returns:
        pd.DataFrame: Columns: DATE and TIME of matching recordings.
    """
    def normalise_df(df):
        "Keep date and time columns and ensure there's no duplicates."
        relevant_cols = ['DATE', 'TIME']
        df = df[relevant_cols]
        df = df.drop_duplicates(subset=['TIME'])
        return df

    mic1_df = pd.read_csv(mic1_summary_path)
    mic2_df = pd.read_csv(mic2_summary_path)
    mic1_df = normalise_df(mic1_df)
    mic2_df = normalise_df(mic2_df)
    matching_times = pd.merge(mic1_df, mic2_df, how='inner', on=['DATE', 'TIME'])
    return matching_times


def link_recordings(data_dict: dict,
                   raw_data_root: str,
                   dataset_root: str,
                   summary_files: list,
                   verbose: bool,
                   file_limit: int = -1):
    """
    
    Creates and populates folders with .wav files matching the times
    in the summary files.

    Args:
        data (dict): .
        summary_paths (list): 
        file_limit (int, optional): = -1. Default 

    Args:
        data_dict (dict): Paths to data organised by year and microphone.
        data_root (str): _description_
        dataset_root (str): _description_
        summary_paths (list): Paths to files containing matching time and dates
            between microphone recordings.
        file_limit (int, optional): Limit number of files copied from
        each folder. -1 copies all. Defaults to -1.
        verbose (bool, optional): Display progress outputs. Defaults to True.

    Returns:
        list: Paths where files have been saved.
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
            raise ValueError('Cannot find '
                             f'{raw_data_root}/{year}/{data_dict.keys()[i]}/{location}')

        dt_df = pd.read_csv(summary)
        for folder in recordings:
            wav_files = os.listdir(folder)
            wav_files.sort()
            if file_limit != -1:
                random.shuffle(wav_files)
                wav_files = wav_files[:file_limit]  # limit here!
            for i, file in enumerate(wav_files):
                date, time = format_file_name(file)
                for _, row in dt_df.iterrows():
                    if row['DATE'] == date and row['TIME'] == time:
                        try:
                            recording_path = os.path.join(folder, file)
                            save_path = f'{dataset_root}{year}/{location}'
                            if verbose:
                                print(f'Linking {recording_path}')
                            # shutil.copy(recording, save_path)
                            recording_paths.add(recording_path)
                        except FileNotFoundError as e:
                            print(str(e))
    return list(recording_paths)


def create_spectrograms(recordings: list,
                        n_fft: int,
                        set_type: str,
                        dataset: str,
                        verbose: bool,
                        target_symbol: str = '-4',
                        hop_length: int = None):
    """
    Create a spectrogram for each matched audio files.
    Parameter dictionaries are saved for all target spectrograms so they
    can be transformed back into audio using pix2pix.utilites.
    

    Args:
        directories (list): _description_
        n_fft (int): _description_
        root (str): _description_
        target_width (int): _description_
        target_symbol (str): _description_
        hop_length (int, optional): _description_. Defaults to None.
        labels (bool, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    print(f'Spectrograms for {set_type} set')
    if hop_length is None:
        hop_length = n_fft // 4
    spectrogram_paths = set()
    for i, wav_file in enumerate(recordings):
        if wav_file.split('.')[-1] != 'wav':
            continue
        try:
            y, sr = librosa.load(wav_file)
            s = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude, phase = librosa.magphase(s)
            s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

            # normalise spectrogram to range (0, 255)
            s_db_norm = 255 * (s_db - s_db.min()) / (s_db.max() - s_db.min())
            s_db_norm = s_db_norm.astype(np.uint8)

            # create path and directory
            recordings = os.path.join(dataset, 'spectrograms', set_type)
            if verbose:
                print(f'Generating spectrogram for {wav_file}, {i+1}/{len(recordings)}')
            os.makedirs(recordings, exist_ok=True)
            # save image
            image = Image.fromarray(s_db_norm)

            # if target_width is not None:
            #     aspect_ratio = image.width / image.height
            #     target_height = int(target_width / aspect_ratio)
            #     image = image.resize((target_width, target_height), Image.LANCZOS)
            
            # raw_data_test/2023_11/SM4/PLI2/PLI2-4_20231208_184900.wav
            # recordings = 'data/spectrograms'
            image.save(f"{recordings}/{wav_file.split('/')[4].replace('.wav', '.png')}")
            spectrogram_paths.add(recordings)

            # create params dict for test set to revert synth spectrograms back to audio
            if set_type == 'test_set':
                params = {
                    'magnitude_real': magnitude.real.tolist(),
                    'magnitude_imag': magnitude.imag.tolist(),
                    'phase_real': phase.real.tolist(),
                    'phase_imag': phase.imag.tolist(),
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'file': wav_file
                }
                params_path = os.path.join(recordings, 'params')
                os.makedirs(params_path, exist_ok=True)
                file_path_params = os.path.join(params_path, wav_file.split('/')[4].replace('.wav', '.json'))
                if verbose:
                    print(f'Saved {file_path_params}')

                with open(file_path_params, 'w') as f:
                    json.dump(params, f)

        except Exception as e:
            print(str(e))
    return list(spectrogram_paths)


def pair_spectrograms(directories: list):
    """
    Pairs the paths to SMMicro and SM4 recordings based on location,
    date and time.

    Args:
        directories (list): List of all folders containing spectrograms.
    """
    def extract_datetime(filename):
        """
        Extracts location and datetime from the recordings filename.
        """
        loc, date, time = filename.split('_')
        datetime = '_'.join([date, time])
        return loc, datetime

    # get independent lists of SMMicro and SM4 spectrograms
    mic1_spectrograms = []
    mic2_spectrograms = []
    for directory in directories:
        files = os.listdir(directory)
        for filename in files:
            if filename == 'params':
                continue
            loc, datetime = extract_datetime(filename)
            if '-4' in filename:
                loc = loc.replace('-4', '')
                mic2_spectrograms.append((loc, datetime, os.path.join(directory, filename)))
            else:
                mic1_spectrograms.append((loc, datetime, os.path.join(directory, filename)))

    # match SMMicro and SM4 spectrograms based on
    # location and datetime in their filenames
    paired_spectrograms = []
    for loc1, dt1, file1 in mic1_spectrograms:
        for loc2, dt2, file2 in mic2_spectrograms:
            if loc1 == loc2 and dt1 == dt2:
                # pass these files straight into stitch images
                paired_spectrograms.append((file1, file2))

    return paired_spectrograms


def stitch_images(paired_spectrogram_paths: list[tuple], dataset_root: str):
    """
    Stiches SMMicro and SM4 paired spectrograms with SMMicro images on the left.
    Matches dataset format for a pix2pix cGAN.

    Args:
        paired_spectrogram_paths (list[tuple]): SMMicro and SM4 pairs.
        dataset_root (str): Root to the dataset.
    """
    def load_spectrogram_as_np_arr(file_path: str):
        image = Image.open(file_path)
        return np.array(image)

    def cross_correlate(spec1: np.array, spec2: np.array):
        """
        Calculates where spec2 best aligns with spec1 by finding the
        x-coordinate with the highest correlation.
        """
        correlation = correlate2d(spec1, spec2, mode='valid')
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
        return x

    def have_same_dimensions(spec1, spec2):
        return spec1.shape == spec2.shape

    os.makedirs(f'{dataset_root}dataset', exist_ok=True)
    separator_width = 0
    separator_colour = (255, 255, 255)
    correlated = False
    for i, (mic1_path, mic2_path) in enumerate(paired_spectrogram_paths):
        # load spectrograms as np arrays
        mic1_spec = load_spectrogram_as_np_arr(mic1_path)
        mic2_spec = load_spectrogram_as_np_arr(mic2_path)

        if not have_same_dimensions(mic1_spec, mic2_spec):
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

        # get dimensions
        mic1_height, mic1_width = mic1_spec.shape
        mic2_height, mic2_width = mic2_spec.shape

        # ensure widths are the same
        min_width = min(mic1_width, mic2_width)
        mic1_spec = mic1_spec[:, :min_width]
        mic2_spec = mic2_spec[:, :min_width]
        if not have_same_dimensions(mic1_spec, mic2_spec):
            raise IndexError('Spectrograms have different dimensions')

        extended_width = mic1_width * 2 + separator_width
        height = mic1_height

        # stiched aligned spectrograms on the same canvas
        # with SMMicro on the left and SM4 on the right
        stitched = Image.new('RGB', (extended_width, height), separator_colour)
        mic1_spec_img = Image.fromarray(mic1_spec)
        mic2_spec_img = Image.fromarray(mic2_spec)
        stitched.paste(mic1_spec_img, (0, 0))
        stitched.paste(mic2_spec_img, (mic1_width + separator_width, 0))
        datetime = mic1_path.split('/')[-1]  # includes '.png' at the end
        if correlated:
            os.makedirs(f'{dataset_root}dataset/correlated', exist_ok=True)
            output_dir = os.path.join(dataset_root, 'dataset', 'correlated', datetime)
        else:
            output_dir = os.path.join(dataset_root, 'dataset', datetime)
        try:
            stitched.save(output_dir)
        except SystemError as se:
            # likely that images are larger than the canvas
            print(f'{str(se)}\nLikely dimensions mismatch'
                          f'Input dimensions (w x h): {mic1_width, mic1_height}\n'
                          f'Target dimensions (w x h):{mic2_width, mic2_height}\n'
                          f'Canvas size (w x h): {stitched.width, stitched.height}')
        except Exception as e:
            print(str(e))
        correlated = False
        print(f'Stitched {i + 1}/{len(paired_spectrogram_paths)} images')


def main():
    raw_data = 'raw_data_test/'
    data = 'data/'
    file_limit = -1  # -1 means all files from all folders, otherwise n files from all folders
    
    # remove hidden files from macOS or Windows systems
    if platform.system() == 'Darwin' or platform.system() == 'Windows':
        remove_hidden_files(raw_data)

    # create a directory to store the dataset in
    os.makedirs(data, exist_ok=True)

    print('Analysing recordings')
    # list metrics for each recording and save them in the dataset folder
    analyse_recordings(raw_data, data, verbose=False)
    print()

    # organise data by year and microphone
    data_dict = create_data_dict(raw_data, data)

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
    matched_summaries = [match_summaries(summary_paths[year], data, verbose=True)
                         for year in summary_paths.keys()]
    print()

    print('Linking recordings')
    # copy matching recordings from raw data to dataset folders
    recordings = [link_recordings(data_dict, raw_data,
                                  data, summary_file,
                                  verbose=True, file_limit=file_limit)
                  for summary_file in matched_summaries]
    all_recordings = [recording for year in recordings for recording in year]  # flatten list of lists
    
    # pair recordings from 
    print()

    print('Creating train/val/test split')
    train, val, test, = train_val_test_split(all_recordings, 0.8)
    
    print(f'Training set: {len(train)} elements')
    print(f'Validation set: {len(val)} elements')
    print(f'Test set: {len(test)} elements')
    print()
    
    print('Training set')
    print(train)
    print('Validation set')
    print(val)

    # TODO pair spectrograms before creating them, then the stitched ones can be saved in the appropriate set
    
    print('Creating spectrograms')
    train_paths = create_spectrograms(train, n_fft=4096,
                                      set_type='train_set', dataset=data, verbose=True)
    
    print('train paths')
    print(train_paths)
    
    val_paths = create_spectrograms(val, n_fft=4096,
                                    set_type='val_set', dataset=data, verbose=True)
    print('val_paths')
    print(val_paths)
    
    test_paths = create_spectrograms(test, n_fft=4096,
                                     set_type='test_set', dataset=data, verbose=True)
    
    print('Pairing spectrograms')
    paired_train = pair_spectrograms(train_paths)
    print('Paired train')
    paired_val = pair_spectrograms(val_paths)
    print('Paired val')
    paired_test = pair_spectrograms(test_paths)
    print('Paired test')
    print()
    

    # # merge all available spectrograms into one list
    # spectrogram_paths = [path for sublist in spectrogram_paths for path in sublist]
    

    print('Stiching images')
    # stich images together to create the dataset
    stitch_images(paired_train, data)
    stitch_images(paired_val, data)
    stitch_images(paired_test, data)


if __name__ == '__main__':
    main()
