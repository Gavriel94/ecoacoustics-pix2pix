import pandas as pd
import shutil
import os


def create_summary(dir_path: str,
                   dest_path: str = None):
    """
    Extracts dates and times from file names to create a new summary file.

    Args:
        dir_path (str): Directory containing WAV files.
        dest_path (str, optional): Save file in specified location.
    """
    folder, year, mic, location = dir_path.split('/')
    folder_path = folder + '/' + year + '/' + mic + '/' + 'summaries'
    # create a directory to store the summaries in
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    files = os.listdir(dir_path)
    # sort files by time and date
    files.sort()
    # map from ints to months
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
    # save dates and times as defined in each wav file name
    dic = {
        'DATE': [],
        'TIME': []
    }
    for i, file in enumerate(files):
        _, date, time = file.split('_')
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
        dic['DATE'].append(out_date)
        dic['TIME'].append(out_time)
        if i == 0:
            first_date = year + month + day
        elif i == len(files) - 1:
            last_date = year + month + day
    df = pd.DataFrame(dic)
    # emulate file name format from other summaries: location, first and last date of recording
    file_name = location + '_Summary_' + first_date + '_' + last_date + '.txt'
    file_path = (folder_path + '/' + file_name)
    print(f'Summary file generated\nSaved in {file_path}\n')
    df.to_csv(file_path, index=False)


def find_matches(sm4_summary_path: str,
                 smmicro_summary_path: str,
                 csv_save_path: str = None):
    """
    Iterates through the summary files to find exact matches between time and dates.§

    Args:
        sm4_summary_path (str): Path to SM4 summary file.
        smmicro_summary_path (str): Path to SMMicro summary file.
        csv_save_path (str, optional): Optionally save results as a CSV file.

    Returns:
        pd.DataFrame: DataFrame with two columns: DATE and TIME.
    """
    relevant_cols = ['DATE', 'TIME']
    # Read SM4 data
    sm4_df = pd.read_csv(sm4_summary_path)
    # Drop unnecessary columns
    sm4_df = sm4_df[relevant_cols]
    # Drop duplicate time stamps
    sm4_df = sm4_df.drop_duplicates(subset=['TIME'])

    smmicro_df = pd.read_csv(smmicro_summary_path)
    smmicro_df = smmicro_df[relevant_cols]
    smmicro_df = smmicro_df.drop_duplicates(subset=['TIME'])

    matching_times = pd.merge(sm4_df, smmicro_df, how='inner', on=['DATE', 'TIME'])
    if csv_save_path is not None:
        if '.csv' not in csv_save_path:
            raise ValueError('csv_save_path must include \'.csv\'')
        matching_times.to_csv(csv_save_path, index=False)
    return matching_times


def create_dataset(timestamps: pd.DataFrame,
                   source_dir: str,
                   minutes_offset: int = 0,
                   months_offset: int = 0,
                   verbose: bool = False):
    """
    Extracts the .WAV files from SM4 or SMMicro folders whose filenames match the dates and times
    listed in `matched_timestamps`. These files are copied into the `dataset/` directory into
    folders with the same file structure as `copy_from`. For example, setting
    `copy_from = data/2023_11/SM4/PLI2` then files will be saved in `dataset/2023_11/SM4/PLI2`.

    Months or minutes are sometimes misaligned from the file name itself,
    for example 2023_11/SM4/PLI3 contains data from 11/23 and 12/23.
    The offset allows checking for times beyond the file name.

    Args:
        matched_timestamps (pd.DataFrame): DataFrame with times and dates.
        copy_from (str): Folder where the WAV files are located.
        minutes_offset (int, optional): Look beyond time stated in file name. Defaults to 0.
        months_offset (int, optional): Look beyond time stated in file name. Defaults to 0.
    """
    def format_time(time: str):
        formatted_time = ''
        hrs, mins, secs = time.split(':')
        mins = str(int(mins) + minutes_offset)
        formatted_time = formatted_time + hrs + mins + secs
        return formatted_time

    def get_day(date: str):
        year, mon, day = date.split('-')
        return day

    folder, date, mic, location = source_dir.split('/')
    date = date.replace('_', '')
    date = str(int(date) + months_offset)
    day = [get_day(date) for date in timestamps['DATE']]
    formatted_times = [format_time(time) for time in timestamps['TIME']]
    file_names = []
    for i, time in enumerate(formatted_times):
        if mic == 'SMMicro':
            file_name = source_dir + '/' + location + '_' + date + day[i] + '_' + time
        elif mic == 'SM4':
            file_name = source_dir + '/' + location + '-4' + '_' + date + day[i] + '_' + time
        file_names.append(file_name)

    for file in file_names:
        file = file + '.wav'
        _, year, _, _, wav_file_name = file.split('/')
        dir_path = 'data/' + year + '/' + mic + '/' + location
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dest_dir = dir_path + '/' + wav_file_name
        try:
            if verbose:
                print(f'{dest_dir}')
            shutil.copy(file, dest_dir)
        except FileNotFoundError as e:
            if verbose:
                print(str(e))
    if verbose:
        print()


def summary_len(path):
    with open(path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def num_features(path):
    """
    Counts the number of recordings from a microphone in a year.

    Args:
        path (str): Path to recordings parent folder. I.e. 'data/2023_11/SM4'
    """
    pli_dirs = [d for d in os.listdir(path)]
    counts = []
    for dir in pli_dirs:
        full_path = f'{path}/{dir}'
        len_dir = len(os.listdir(full_path))
        counts.append(len_dir)
        print(f'{full_path}: {len_dir} files')
    print(f'Total number of files: {sum(counts)}\n')


# # generate missing 2023 SM4 summary files
create_summary('raw_data/2023_11/SM4/PLI2')
create_summary('raw_data/2023_11/SM4/PLI3')

print('Finding matches in 2023 files')
pli2_2023 = find_matches('raw_data/2023_11/SM4/summaries/PLI2_Summary_20231208_20231224.txt',
                         'raw_data/2023_11/SMMicro/summaries/PLI2_Summary_20231128-231219.txt',
                         'data/2023_11/summaries/PLI2_2023_matching_summaries.csv')

pli3_2023 = find_matches('raw_data/2023_11/SM4/summaries/PLI3_Summary_20231128_20231223.txt',
                         'raw_data/2023_11/SMMicro/summaries/PLI3_Summary_20231128-231219.txt',
                         'data/2023_11/summaries/PLI3_2023_matching_summaries.csv')

pli2_23_sm_ln = summary_len('data/2023_11/summaries/PLI2_2023_matching_summaries.csv')
pli3_23_sm_ln = summary_len('data/2023_11/summaries/PLI3_2023_matching_summaries.csv')

print(f'Matching times and dates in PLI2: {pli2_23_sm_ln}')
print(f'Matching times and dates in PLI3: {pli3_23_sm_ln}')
print(f'Total: {pli2_23_sm_ln + pli3_23_sm_ln}')
print()

print('Finding matches in 2024 files')
pli1_2024 = find_matches('raw_data/2024_03/SM4/summaries/PLI1-4_A_Summary_202403.txt',
                         'raw_data/2024_03/SMMicro/summaries/PLI1_Summary_20240316_20240424.txt',
                         'data/2024_03/summaries/PLI1_2024_matching_summaries.csv')

pli2_2024 = find_matches('raw_data/2024_03/SM4/summaries/PLI2-4_A_Summary_202403.txt',
                         'raw_data/2024_03/SMMicro/summaries/PLI2_Summary_20240316_20240424.txt',
                         'data/2024_03/summaries/PLI2_2024_matching_summaries.csv')

pli3_2024 = find_matches('raw_data/2024_03/SM4/summaries/PLI3-4_A_Summary_202403.txt',
                         'raw_data/2024_03/SMMicro/summaries/PLI3_Summary_20240316_20240424.txt',
                         'data/2024_03/summaries/PLI3_2024_matching_summaries.csv')

pli1_24_sm_ln = summary_len('data/2024_03/summaries/PLI1_2024_matching_summaries.csv')
pli2_24_sm_ln = summary_len('data/2024_03/summaries/PLI2_2024_matching_summaries.csv')
pli3_24_sm_ln = summary_len('data/2024_03/summaries/PLI3_2024_matching_summaries.csv')

print(f'Matching times and dates in PLI1: {pli1_24_sm_ln}')
print(f'Matching times and dates in PLI2: {pli2_24_sm_ln}')
print(f'Matching times and dates in PLI3: {pli3_24_sm_ln}')
print(f'Total: {pli1_24_sm_ln + pli2_24_sm_ln + pli3_24_sm_ln}')
print()

# recordings are from November and December so running through each location
# twice with an offset
print('Copying 2023 files')
create_dataset(timestamps=pli2_2023,
               source_dir='raw_data/2023_11/SM4/PLI2',
               months_offset=0)

create_dataset(timestamps=pli2_2023,
               source_dir='raw_data/2023_11/SMMicro/PLI2',
               months_offset=0)

create_dataset(timestamps=pli3_2023,
               source_dir='raw_data/2023_11/SM4/PLI3',
               months_offset=0)

create_dataset(timestamps=pli3_2023,
               source_dir='raw_data/2023_11/SMMicro/PLI3',
               months_offset=0)

create_dataset(timestamps=pli2_2023,
               source_dir='raw_data/2023_11/SM4/PLI2',
               months_offset=1)

create_dataset(timestamps=pli2_2023,
               source_dir='raw_data/2023_11/SMMicro/PLI2',
               months_offset=1)

create_dataset(timestamps=pli3_2023,
               source_dir='raw_data/2023_11/SM4/PLI3',
               months_offset=1)

create_dataset(timestamps=pli3_2023,
               source_dir='raw_data/2023_11/SMMicro/PLI3',
               months_offset=1)

# SM4 file names are sometimes 1 minute less than the time listed in the summary so
# running through each directory twice with the offset
print('Copying 2024 files')
print()
create_dataset(timestamps=pli1_2024,
               source_dir='raw_data/2024_03/SM4/PLI1',
               minutes_offset=-1)

create_dataset(timestamps=pli1_2024,
               source_dir='raw_data/2024_03/SM4/PLI1',
               minutes_offset=0)

create_dataset(timestamps=pli1_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI1',
               minutes_offset=-1)

create_dataset(timestamps=pli1_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI1',
               minutes_offset=0)

create_dataset(timestamps=pli2_2024,
               source_dir='raw_data/2024_03/SM4/PLI2',
               minutes_offset=-1)

create_dataset(timestamps=pli2_2024,
               source_dir='raw_data/2024_03/SM4/PLI2',
               minutes_offset=0)

create_dataset(timestamps=pli2_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI2',
               minutes_offset=-1)

create_dataset(timestamps=pli2_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI2',
               minutes_offset=0)

create_dataset(timestamps=pli3_2024,
               source_dir='raw_data/2024_03/SM4/PLI3',
               minutes_offset=-1)

create_dataset(timestamps=pli3_2024,
               source_dir='raw_data/2024_03/SM4/PLI3',
               minutes_offset=0)

create_dataset(timestamps=pli3_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI3',
               minutes_offset=-1)

create_dataset(timestamps=pli3_2024,
               source_dir='raw_data/2024_03/SMMicro/PLI3',
               minutes_offset=0)

# Count the number of total recordings
num_features('data/2023_11/SM4')
num_features('data/2023_11/SMMicro')
num_features('data/2024_03/SM4')
num_features('data/2024_03/SMMicro')
