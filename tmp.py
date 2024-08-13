import os
import config
import pandas as pd
import utilities as utils
from datetime import datetime
import random


def format_datetime(date_str):
    """
    Formats date in format 'YYYYMMDD' to 'YYYY-MM-DD'.
    """
    try:
        date = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
        return datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        print(f'File name incorrectly formatted: {date_str}')


def match_audio_summary(summaries_root, audio_root, save: bool):
    """
    Uses the audios filename to search through all summary file and
    return its specific row of values.

    Args:
        summary_files (_type_): _description_
        generated_audio (_type_): _description_
        summaries_root (_type_): _description_
    """
    # find date in range of two dates in summaries path name
    # if in range, iterate through summary file, find matching datetime and return lat/lon
    audio_files = [file for file in os.listdir(audio_root) if file.endswith('.wav')]
    summary_files = os.listdir(summaries_root)

    rows = []
    row_count = 0
    for audio in audio_files:
        g_loc, g_date, g_time = audio.split('_')
        g_datetime = format_datetime(g_date)
        for summary in summary_files:
            s_loc, _, s_start_date, s_end_date = summary.replace('.txt', '').split('_')

            s_start_datetime = format_datetime(s_start_date)
            s_end_datetime = format_datetime(s_end_date)
            # match audio with the date and location of the summary file
            if s_start_datetime <= g_datetime <= s_end_datetime and s_loc == g_loc:
                summary_df = pd.read_csv(os.path.join(summaries_root, summary), delimiter=',')
                gen_audio_name = audio.replace('.wav', '')
                gen_audio_tl = utils.filename_to_datetime(gen_audio_name)

                data = summary_df[(summary_df['DATE'].astype(str)
                                   == str(gen_audio_tl[0]))
                                  & (summary_df['TIME'].astype(str)
                                     == str(gen_audio_tl[1]))]
                if not data.empty:
                    row_count += 1
                    rows.append(data)
                else:
                    raise ValueError(f'Empty row for {audio} in summary file')

    # combine list of single-row DataFrames into one df
    combined_rows = pd.concat(rows, ignore_index=True)
    if save:
        combined_rows.to_csv(os.path.join(summaries_root, 'audio_matched.csv'))
    return combined_rows


def bird_net_analysis(audio):
    


def main():
    # ensure no metadata files exist
    utils.remove_hidden_files(config.RAW_DATA_ROOT)
    utils.remove_hidden_files(config.DATASET_ROOT)

    # list of full summaries
    summaries_root = os.path.join(config.RAW_DATA_ROOT, 'full_summaries')
    # audio generated from test data
    gen_audio_root = os.path.join(config.DATASET_ROOT, 'evaluate', 'audio')

    audio_matched_path = os.path.join(summaries_root, 'audio_matched.csv')
    if not os.path.exists(audio_matched_path):
        # generates a df of summary data for all audio files
        audio_df = match_audio_summary(summaries_root, gen_audio_root, save=True)
    else:
        audio_df = pd.read_csv(audio_matched_path)

    # get a random generated audio
    generated_audio = random.choice(os.listdir(gen_audio_root))
    raw_audio = utils.get_raw_audio(generated_audio, config.RAW_DATA_ROOT, 'SM4', '-4')

    date, time = utils.filename_to_datetime(generated_audio)
    # get recordings latitude and longitude from summary file
    lat_lon = audio_df[(audio_df['DATE'] == date)
                       & (audio_df['TIME'] == time)][['LAT', 'LON']].iloc[0]

    print(generated_audio, raw_audio)


    # bird_net_analysis()








    # bird_net_analysis(summaries, test_files)
    # print(data['LAT'])
    # lat = data['LAT'].item()
    # lon = data['LAT'].item()
    # print(lat, lon)

if __name__ == '__main__':
    main()
