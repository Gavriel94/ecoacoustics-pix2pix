import os
import config
import pandas as pd
import utilities as utils
from datetime import datetime
import random
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import json


def format_datetime(date_str):
    """
    Formats date in format 'YYYYMMDD' to 'YYYY-MM-DD'.
    """
    try:
        date = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
        return datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        print(f'File name incorrectly formatted: {date_str}')


def match_audio_summary(summaries_root, audio_root):
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
    return combined_rows


def birdnet_analysis(summaries_root, gen_audio_root, audio_samples=[], num_samples=5):
    """
    Pair and run generated and raw recordings through BirdNet.
    Produces JSON files for inspection.

    Args:
        summaries_root (_type_): _description_
        gen_audio_root (_type_): _description_
        num_samples (_type_): _description_

    Raises:
        ValueError: _description_
    """
    if num_samples <= 0:
        raise ValueError('Must analyse at least 1 sample.')
    # generates a df of summary data for all audio files
    audio_df = match_audio_summary(summaries_root, gen_audio_root, save=True)

    if len(audio_samples) == 0:
        audio_samples = random.sample(os.listdir(gen_audio_root), num_samples)
    months = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
    }
    analysis_root = os.path.join(config.DATASET_ROOT, 'evaluate', 'birdnet_results')
    os.makedirs(analysis_root, exist_ok=True)
    analyzer = Analyzer()
    for audio in audio_samples:
        raw_audio = utils.get_raw_audio(audio, config.RAW_DATA_ROOT, 'SM4', '-4')
        date, time = utils.filename_to_datetime(audio)
        year, month, day = date.split('-')
        month = months[month]
        # get recordings latitude and longitude from summary file
        lat_lon = audio_df[(audio_df['DATE'] == date)
                           & (audio_df['TIME'] == time)][['LAT', 'LON']].iloc[0]

        gen_recording = Recording(
            analyzer,
            os.path.join(gen_audio_root, audio),
            lat_lon['LAT'],
            lon=lat_lon['LON'],
            date=datetime(year=int(year), month=int(month), day=int(day)),
            min_conf=0.25
        )
        gen_recording.analyze()

        raw_recording = Recording(
            analyzer,
            raw_audio,
            lat_lon['LAT'],
            lon=lat_lon['LON'],
            date=datetime(year=int(year), month=int(month), day=int(day)),
            min_conf=0.25
        )
        raw_recording.analyze()
        save = os.path.join(analysis_root, audio.replace('.wav', ''))
        os.makedirs(save)
        with open(os.path.join(save, 'gen.json'), 'w') as f:
            json.dump(gen_recording.detections, f)
        with open(os.path.join(save, 'raw.json'), 'w') as f:
            json.dump(raw_recording.detections, f)
