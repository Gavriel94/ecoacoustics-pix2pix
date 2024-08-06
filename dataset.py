import pandas as pd
import os
import librosa
import numpy as np
import platform
import shutil
import json
from PIL import Image
from scipy.signal import correlate2d

# from pix2pix.config import setup_logging
from audio_analysis import analyse_recordings
import logging
import pix2pix.utilities as utils

# setup_logging()

logging.basicConfig(format='%(asctime)s:%(message)s',
                    level=logging.INFO,
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
                    logging.error(f'Error removing {file_path}: {e}.\n'
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
            logging.info(f'Creating summary for {dest}')
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
    logging.info(f'Summary file generated\nSaved in {file_path}\n')
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
    def create_save_dir(data_to):
        """
        Strips the year and location from a full summary path to generate
        an appropriate save directory.
        """
        _, year, _, _, loc = sm4_path.split('/')
        loc = loc.split('_')[0]
        path = os.path.join(data_to, year, loc)
        os.makedirs(path, exist_ok=True)
        return path

    # sort and collect child paths for each summary directory by location
    loc_sorted = [sorted(os.listdir(file)) for file in summary_dir]
    # pair summary files from corresponding microphones
    paired = list(zip(*loc_sorted))
    # prepend the full path information to the summary file
    paired_paths = [(os.path.join(summary_dir[0], pair[0]),
                     os.path.join(summary_dir[1], pair[1])) for pair in paired]
    summaries = []
    for pair in paired_paths:
        sm4_path, smmicro_path = pair
        matches = match_times(sm4_path, smmicro_path)
        dir_path = create_save_dir(data_to)
        save_path = dir_path + '/summary.csv'
        matches.to_csv(save_path)
        summaries.append(save_path)
        if verbose:
            logging.info(f'Saved matches from {pair} to {save_path}')
    return summaries


def match_times(sm4_summary_path: str, smmicro_summary_path: str):
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

    sm4_df = pd.read_csv(sm4_summary_path)
    smm_df = pd.read_csv(smmicro_summary_path)
    sm4_df = normalise_df(sm4_df)
    smm_df = normalise_df(smm_df)
    matching_times = pd.merge(sm4_df, smm_df, how='inner', on=['DATE', 'TIME'])
    return matching_times


def get_recordings(data_dict: dict,
                   data_root: str,
                   dataset_root: str,
                   summary_paths: list,
                   verbose: bool = True):
    """
    Creates and populates folders with .wav files matching the times
    in the summary files.

    Args:
        data (dict): Paths to data organised by year and microphone.
        summary_paths (list): Paths to files containing matching time and dates
            between microphone recordings.

    Returns:
        list: Paths where files have been saved.
    """
    save_paths = set()
    for i, summary in enumerate(summary_paths):
        _, year, location, _ = summary.split('/')
        # paths to the appropriate recording location for each microphone
        recordings = [path for sublist in list(data_dict[year].values())
                      for path in sublist if location in path]

        # ensure folders containing recordings exist
        if len(recordings) <= 0:
            raise ValueError('Cannot find '
                             f'{data_root}/{year}/{data_dict.keys()[i]}/{location}')

        dt_df = pd.read_csv(summary)
        for folder in recordings:
            wav_files = os.listdir(folder)
            wav_files.sort()
            for i, file in enumerate(wav_files):
                date, time = format_file_name(file)
                for _, row in dt_df.iterrows():
                    if row['DATE'] == date and row['TIME'] == time:
                        try:
                            recording = os.path.join(folder, file)
                            save_path = f'{dataset_root}{year}/{location}'
                            if verbose:
                                logging.info(f'Copying {recording} to {save_path}')
                            shutil.copy(recording, save_path)
                            save_paths.add(save_path)
                        except FileNotFoundError as e:
                            logging.error(str(e))
    return list(save_paths)


def create_spectrograms(directories: list,
                        n_fft: int,
                        root: str,
                        target_width: int = 512,
                        hop_length: int = None,
                        labels: bool = None,
                        verbose: bool = False):
    """
    Create spectrograms for all .wav files in a folder.

    Args:
        directories (list): Paths to folders containing .wav files.
        n_fft (int): Number of fast Fourier transform points.
        hop_length (int, optional): Number of samples in each frame.
            Default is n_fft // 4.
        verbose (bool, optional): Display progress. Defaults to False.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    save_paths = set()
    for directory in directories:
        files = os.listdir(directory)
        for i, file in enumerate(files):
            if file.split('.')[-1] == 'csv':
                continue
            if verbose:
                logging.info(f'Generating spectrogram for {file}, '
                             f'{i + 1}/{len(files)} files')
            path = os.path.join(directory, file)
            try:
                y, sr = librosa.load(path)
                s = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

                # normalise spectrogram to range (0, 255)
                s_db_norm = 255 * (s_db - s_db.min()) / (s_db.max() - s_db.min())
                s_db_norm = s_db_norm.astype(np.uint8)

                # create path and directory
                save_path = directory.replace(f'{root}',
                                              f'{root}spectrograms/')
                os.makedirs(save_path, exist_ok=True)
                # save image
                image = Image.fromarray(s_db_norm)
                image.save(f"{save_path}/{file.replace('.wav', '.png')}")
                save_paths.add(save_path)
            except Exception as e:
                logging.error(str(e))
    return list(save_paths)


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
    smmicro_spectrograms = []
    sm4_spectrograms = []
    for directory in directories:
        files = os.listdir(directory)
        for filename in files:
            loc, datetime = extract_datetime(filename)
            if '-4' in filename:
                loc = loc.replace('-4', '')
                sm4_spectrograms.append((loc, datetime, os.path.join(directory, filename)))
            else:
                smmicro_spectrograms.append((loc, datetime, os.path.join(directory, filename)))

    # match SMMicro and SM4 spectrograms based on
    # location and datetime in their filenames
    paired_spectrograms = []
    for loc1, dt1, file1 in smmicro_spectrograms:
        for loc2, dt2, file2 in sm4_spectrograms:
            if loc1 == loc2 and dt1 == dt2:
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
    for i, (smmicro_path, sm4_path) in enumerate(paired_spectrogram_paths):
        # load spectrograms as np arrays
        smm_spec = load_spectrogram_as_np_arr(smmicro_path)
        sm4_spec = load_spectrogram_as_np_arr(sm4_path)
        # * DEBUGGING
        # print('saving images in tmp')
        # utils.save_img_arr_in_tmp(smm_spec, smmicro_path)
        # utils.save_img_arr_in_tmp(sm4_spec, sm4_path)
        if not have_same_dimensions(smm_spec, sm4_spec):
            # align them using cross correlation
            offset = cross_correlate(smm_spec, sm4_spec)
            correlated = True
            # trim spectrograms so they're the same length
            if offset > 0:
                smm_spec = smm_spec[:, offset:]
                sm4_spec = sm4_spec[:, :smm_spec.shape[1]]
            else:
                sm4_spec = sm4_spec[:, -offset:]
                smm_spec = smm_spec[:, :sm4_spec.shape[1]]
            utils.save_img_arr_in_tmp(smm_spec, smmicro_path, 'smm')
            utils.save_img_arr_in_tmp(sm4_spec, sm4_path, 'sm4')

        # get dimensions
        smmicro_height, smmicro_width = smm_spec.shape
        sm4_height, sm4_width = sm4_spec.shape

        # ensure widths are the same
        min_width = min(smmicro_width, sm4_width)
        smm_spec = smm_spec[:, :min_width]
        sm4_spec = sm4_spec[:, :min_width]

        if not have_same_dimensions(smm_spec, sm4_spec):
            raise IndexError('Spectrograms have different dimensions')

        extended_width = smmicro_width * 2 + separator_width
        height = smmicro_height

        # stiched aligned spectrograms on the same canvas
        # with SMMicro on the left and SM4 on the right
        stitched = Image.new('RGB', (extended_width, height), separator_colour)
        smm_spec_img = Image.fromarray(smm_spec)
        sm4_spec_img = Image.fromarray(sm4_spec)
        stitched.paste(smm_spec_img, (0, 0))
        stitched.paste(sm4_spec_img, (smmicro_width + separator_width, 0))
        datetime = smmicro_path.split('/')[-1]  # includes '.png' at the end
        if correlated:
            os.makedirs(f'{dataset_root}dataset/correlated', exist_ok=True)
            output_dir = os.path.join(dataset_root, 'dataset', 'correlated', datetime)
        else:
            output_dir = os.path.join(dataset_root, 'dataset', datetime)
        try:
            stitched.save(output_dir)
        except SystemError as se:
            # likely that images are larger than the canvas
            logging.error(f'{str(se)}\nLikely dimensions mismatch'
                          f'Input dimensions (w x h): {smmicro_width, smmicro_height}\n'
                          f'Target dimensions (w x h):{sm4_width, sm4_height}\n'
                          f'Canvas size (w x h): {stitched.width, stitched.height}')
        except Exception as e:
            logging.error(str(e))
        correlated = False
        logging.info(f'Stitched {i + 1}/{len(paired_spectrogram_paths)} images')


def create_dataset(data_root: str,
                   dataset_root: str,
                   analysis: bool,
                   matched_summaries: list | None = None,
                   copied_recordings: list | None = None,
                   spectrogram_paths: list | None = None,
                   verbose: bool = True):
    # remove hidden files from macOS or Windows systems
    if platform.system() == 'Darwin' or platform.system() == 'Windows':
        remove_hidden_files(data_root)

    # create a directory to store the dataset in
    os.makedirs(dataset_root, exist_ok=True)

    if analysis:
        logging.debug('Analysing recordings')
        # list metrics for each recording and save them in the dataset folder
        analyse_recordings(data_root, dataset_root, verbose)

    # organise data by year and microphone
    data_dict = create_data_dict(data_root, dataset_root)

    # get summary files by date
    summary_paths = {}
    for date, mics in data_dict.items():
        for mic, paths in mics.items():
            for path in paths:
                if 'summaries' in path:
                    if date not in summary_paths:
                        summary_paths[date] = []
                    summary_paths[date].append(path)

    if matched_summaries is None:
        if summary_paths is None:
            raise UnboundLocalError('Summary paths were not generated.')
        # generate CSV files listing matched recordings from data in summary files
        logging.debug('Matching summaries')
        matched_summaries = [match_summaries(summary_paths[year], dataset_root, verbose)
                             for year in summary_paths.keys()]

    if copied_recordings is None:
        if not matched_summaries:
            raise UnboundLocalError('No matched summaries')
        logging.debug('Copying recordings')
        # copy matching recordings from raw data to dataset folders
        copied_recordings = [get_recordings(data_dict, data_root,
                                            dataset_root, summary_file)
                             for summary_file in matched_summaries]

    if spectrogram_paths is None:
        if not copied_recordings:
            raise UnboundLocalError('No paths to recordings')
        logging.debug('Creating spectrograms')
        spectrogram_paths = [create_spectrograms(audio, n_fft=4096,
                                                 root=dataset_root, verbose=verbose)
                             for audio in copied_recordings]

    # # merge all available spectrograms into one list
    spectrogram_paths = [path for sublist in spectrogram_paths for path in sublist]

    # find matching pairs by pathname
    paired_spectrograms = pair_spectrograms(spectrogram_paths)

    logging.debug('Stiching images')
    # stich images together to create the dataset
    stitch_images(paired_spectrograms, dataset_root)
