import pandas as pd
import os
import platform
import shutil
import json


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
                    print(f'Error removing {file_path}: {e}')
                    print('Perhaps try manually removing.')


def create_data_dict(data_from: str, data_to: str, save_json=True):
    """
    Creates a dictionary of paths to all recordings organised by year and
    microphone. Is able to save a JSON file is saved in the dataset directory.

    Args:
        data_from (str): Path of source data.
        data_to (str): Path where dataset will be stored.
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
        # PLI2_20240106_014700.wav
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


def format_file_name(file_name):
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
    out_date = year + '-' + months[month] + '-' + day
    # emulate time format from other summaries
    out_time = hours + ':' + mins + ':' + secs
    return out_date, out_time


def match_summaries(summary_dir: list, data_to: str):
    """
    Finds matching times from different microphones for each location-based summary file.

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
        sm4_path = pair[0]
        smmicro_path = pair[1]
        matches = match_times(sm4_path, smmicro_path)
        dir_path = create_save_dir(data_to)
        save_path = dir_path + '/summary.csv'
        matches.to_csv(save_path)
        summaries.append(save_path)
    return summaries


def match_times(sm4_summary_path: str, smmicro_summary_path: str):
    """
    Inspects summary files to find recordings taken at the same time and date.

    Args:
        sm4_summary_path (str): Path to SM4 summary file.
        smmicro_summary_path (str): Path to SMMicro summary file.

    Returns:
        pd.DataFrame: Columns: DATE and TIME where recordings match.
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


def get_recordings(data: dict, summary_paths: str):
    """
    Creates and populates folders with .wav files matching the times
    in the summary files.

    Args:
        data (dict): Paths to data organised by year and microphone.
        summary_paths (str): _description_
    """
    for i, summary in enumerate(summary_paths):
        _, year, location, _ = summary.split('/')
        # paths to the appropriate recording location for each microphone
        recordings = [path for sublist in list(data[year].values())
                      for path in sublist if location in path]

        # ensure folders containing recordings exist
        if len(recordings) <= 0:
            raise ValueError('Cannot find a folder containing recordings.\n'
                             f'Looking for {DATA_ROOT}/{year}/{data.keys()[i]}/{location}')

        dt_df = pd.read_csv(summary)
        for folder in recordings:
            wav_files = os.listdir(folder)
            wav_files.sort()
            for i, file in enumerate(wav_files):
                date, time = format_file_name(file)
                for _, row in dt_df.iterrows():
                    if row['DATE'] == date and row['TIME'] == time:
                        try:
                            print('Match found')
                            recording = os.path.join(folder, file)
                            save_path = f'{DATASET_ROOT}/{year}/{location}'
                            print(f'Copying {recording} to {save_path}')
                            shutil.copy(recording, save_path)
                        except FileNotFoundError as e:
                            print(str(e))


def create_dataset():
    # remove hidden files from macOS or Windows systems
    if platform.system() == 'Darwin' or platform.system() == 'Windows':
        remove_hidden_files(DATA_ROOT)

    # create a directory to store the dataset in
    os.mkdir(DATASET_ROOT)

    # organise the data by year and microphone
    data_dict = create_data_dict(DATA_ROOT, DATASET_ROOT)

    # get paths to all data in each year, organised by microphone
    paths_2023 = get_paths(data_dict, '2023_11')
    paths_2024 = get_paths(data_dict, '2024_03')

    # list of paths to the summary directories for all microphones
    summ_dir23 = [path for sublist in list(paths_2023.values())
                  for path in sublist if 'summaries' in path]
    summ_dir24 = [path for sublist in list(paths_2024.values())
                  for path in sublist if 'summaries' in path]

    # find matches between summary files, save them and retain the path
    summary_files_23 = match_summaries(summ_dir23, DATASET_ROOT)
    summary_files_24 = match_summaries(summ_dir24, DATASET_ROOT)

    get_recordings(data_dict, summary_files_23)
    get_recordings(data_dict, summary_files_24)


DATA_ROOT = 'raw_data/'
DATASET_ROOT = 'data/'

create_dataset()
