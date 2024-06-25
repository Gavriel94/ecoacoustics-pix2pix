import pandas as pd
import shutil
import os


def find_matches(sm4_summary_path: str,
                 smmicro_summary_path: str,
                 save_path: str = None):
    """
    Identifies which dates and times are exact matches in two summary files.
    """
    relevant_cols = ['DATE', 'TIME', 'LAT', 'LON']
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
    if save_path is not None:
        matching_times.to_csv(save_path, index=False)
    return matching_times


def create_dataset(matched_timestamps: pd.DataFrame, destination: str, mins_offset: int = 0):
    """
    Iterates through the df containing matching summaries and creates a list of matching SMMicro
    and SM4 paths. Copies these files to `wav_file_path` to create the dataset.
    Times in the 2024 file names are 1 minute out of the times listed in the summaries hence the
    mins_offset parameter.

    Args:
        matched_timestamps (pd.DataFrame): Df containing the date and times of matching summaries.
        destination (str): Where to save the WAV files.
        mins_offset (int, optional): Align file name with time listed in summary. Defaults to 0.
    """
    def format_time(time: str):
        formatted_time = ''
        hrs, mins, secs = time.split(':')
        mins = int(mins) - mins_offset
        formatted_time = formatted_time + hrs + str(mins) + secs
        return formatted_time

    def get_day(date: str):
        year, mon, day = date.split('-')
        return day

    folder, date, mic, location = destination.split('/')

    date = date.replace('_', '')
    day = [get_day(date) for date in matched_timestamps['DATE']]
    formatted_times = [format_time(time) for time in matched_timestamps['TIME']]
    file_names = []
    for i, time in enumerate(formatted_times):
        if mic == 'SMMicro':
            file_name = destination + '/' + location + '_' + date + day[i] + '_' + time
        elif mic == 'SM4':
            file_name = destination + '/' + location + '-4' + '_' + date + day[i] + '_' + time
        file_names.append(file_name)

    for file in file_names:
        file = file + '.wav'
        _, year, _, _, wav_file_name = file.split('/')
        destination = 'data/dataset/' + year + '/' + location + '/' + wav_file_name
        try:
            shutil.copy(file, destination)
        except FileNotFoundError as e:
            print(str(e))


pli1 = find_matches('data/2024_03/SM4/summaries/PLI1-4_A_Summary_202403.txt',
                    'data/2024_03/SMMicro/summaries/PLI1_Summary_20240316_20240424.txt',
                    'data/analysis/summary_matches/2024_03/PLI1.csv')
pli2 = find_matches('data/2024_03/SM4/summaries/PLI2-4_A_Summary_202403.txt',
                    'data/2024_03/SMMicro/summaries/PLI2_Summary_20240316_20240424.txt',
                    'data/analysis/summary_matches/2024_03/PLI2.csv')
pli3 = find_matches('data/2024_03/SM4/summaries/PLI3-4_A_Summary_202403.txt',
                    'data/2024_03/SMMicro/summaries/PLI3_Summary_20240316_20240424.txt',
                    'data/analysis/summary_matches/2024_03/PLI3.csv')

create_dataset(pli1, 'data/2024_03/SMMicro/PLI1', mins_offset=1)
create_dataset(pli1, 'data/2024_03/SM4/PLI1', mins_offset=1)

create_dataset(pli2, 'data/2024_03/SMMicro/PLI2', mins_offset=1)
create_dataset(pli2, 'data/2024_03/SM4/PLI2', mins_offset=1)

create_dataset(pli3, 'data/2024_03/SMMicro/PLI3', mins_offset=1)
create_dataset(pli3, 'data/2024_03/SM4/PLI3', mins_offset=1)

len_pli1 = len(os.listdir('data/2024_03/SM4/PLI1'))
len_pli2 = len(os.listdir('data/2024_03/SM4/PLI2'))
len_pli3 = len(os.listdir('data/2024_03/SM4/PLI3'))
print(f'2024_03 PLI1 data samples: {len_pli1}')
print(f'2024_03 PLI2 data samples: {len_pli2}')
print(f'2024_03 PLI3 data samples: {len_pli3}')
print('Total 2024 samples:', len_pli1 + len_pli2 + len_pli3)
