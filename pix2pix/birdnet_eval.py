import os
import random
from collections import defaultdict
from datetime import datetime

import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

import config
import utilities as utils


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
    Returns an audio files summary details.

    Args:
        summaries_root (str): Root to summary files
        audio_root (str): Root to data.
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
            if summary.split('.')[0] == 'full_summaries':
                continue
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


def birdnet_analysis(summaries_root, gen_audio_root, num_run, audio_samples=[], random_samples=5):
    """
    Runs BirdNet on a collection of generated and real recordings.
    A collection of audio samples can be analysed, or samples can be selected
    at random.

    The data is saved in the evaluation folder.

    Args:
        summaries_root (str): Root to directory containing full summary files.
        gen_audio_root (str): Root of directory containing synthetic audio.
        audio_samples (list): One or more audio samples.
        num_samples (int): Number of random samples to use.

    Raises:
        ValueError: No audio samples.
    """
    full_summaries_path = os.path.join(summaries_root, 'full_summaries.csv')
    # generates a df of summary data for all audio files
    audio_df = match_audio_summary(summaries_root, gen_audio_root)
    audio_df.to_csv(full_summaries_path)

    if len(audio_samples) < 1:
        # no audio selected, get all audio samples
        if random_samples == -1:
            audio_samples = os.listdir(gen_audio_root)
        else:
            audio_samples = random.sample(os.listdir(gen_audio_root), random_samples)

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
    analyzer = Analyzer()
    results = []
    for audio in audio_samples:
        # get matching raw audio
        raw_audio = utils.get_raw_audio(audio)
        # transform filename to match summaries
        date, time = utils.filename_to_datetime(audio)
        year, month, day = date.split('-')
        month = months[month]

        try:
            # get recordings latitude and longitude from summary file
            lat_lon = audio_df[(audio_df['DATE'].astype(str) == date)
                               & (audio_df['TIME'] == time)][['LAT', 'LON']].iloc[0]
        except IndexError as e:
            print(str(e))
            print(f'cannot analyse {gen_audio_root}')
            continue
        # run generated audio through BirdNet
        gen_recording = Recording(
            analyzer,
            os.path.join(gen_audio_root, audio),
            lat_lon['LAT'],
            lon=lat_lon['LON'],
            date=datetime(year=int(year), month=int(month), day=int(day)),
            min_conf=0.25
        )
        gen_recording.analyze()

        # run raw audio through BirdNet
        raw_recording = Recording(
            analyzer,
            raw_audio,
            lat_lon['LAT'],
            lon=lat_lon['LON'],
            date=datetime(year=int(year), month=int(month), day=int(day)),
            min_conf=0.25
        )
        raw_recording.analyze()

        # create a database for the results
        gen_dict = defaultdict(list)
        raw_dict = defaultdict(list)
        for det in gen_recording.detections:
            gen_dict[det['common_name']].append(det)

        for det in raw_recording.detections:
            raw_dict[det['common_name']].append(det)

        all_species = set(gen_dict.keys()).union(set(raw_dict.keys()))

        for species in all_species:
            gen_detections = gen_dict.get(species, [])
            raw_detections = raw_dict.get(species, [])

            for g_det in gen_detections:
                matched = False
                for r_det in raw_detections:
                    if g_det['start_time'] == r_det['start_time'] \
                            and g_det['end_time'] == r_det['end_time']:
                        results.append({
                            'audio_file': audio,
                            'species': species,
                            'gen_start_time': g_det['start_time'],
                            'gen_end_time': g_det['end_time'],
                            'gen_confidence': g_det['confidence'],
                            'raw_start_time': r_det['start_time'],
                            'raw_end_time': r_det['end_time'],
                            'raw_confidence': r_det['confidence'],
                            'match': True
                        })
                        matched = True
                        break
                if not matched:
                    results.append({
                        'audio_file': audio,
                        'species': species,
                        'gen_start_time': g_det['start_time'],
                        'gen_end_time': g_det['end_time'],
                        'gen_confidence': g_det['confidence'],
                        'raw_start_time': None,
                        'raw_end_time': None,
                        'raw_confidence': None,
                        'match': False
                    })

            for r_det in raw_detections:
                if not any(r_det['start_time'] == res['raw_start_time'] and
                           r_det['end_time'] == res['raw_end_time'] for res in results):
                    results.append({
                        'audio_file': audio,
                        'species': species,
                        'gen_start_time': None,
                        'gen_end_time': None,
                        'gen_confidence': None,
                        'raw_start_time': r_det['start_time'],
                        'raw_end_time': r_det['end_time'],
                        'raw_confidence': r_det['confidence'],
                        'match': False
                    })

    df = pd.DataFrame(results)
    os.makedirs(os.path.join(config.DATASET_ROOT, 'evaluation', f'run_{num_run}'), exist_ok=True)
    df.to_csv(os.path.join(config.DATASET_ROOT, 'evaluation', f'run_{num_run}', 'birdnet_analysis.csv'))
