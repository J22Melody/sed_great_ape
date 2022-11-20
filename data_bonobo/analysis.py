from __future__ import annotations
import pandas as pd
import numpy as np
import librosa
import os
import csv
import re
from pathlib import Path
from pprint import pprint


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

WAV_PATH = './raw/bonobo calls'
ANNOTATION_PATH = './raw/metadata.csv'

df_raw = pd.read_csv(ANNOTATION_PATH, keep_default_na=False)

data = []
call_map = {}
annotation_duration = 0

wavpaths = Path(WAV_PATH).rglob('*.wav')

def convert_time(input):
    mm, ss = input.split(':')
    return int(mm) * 60 + float(ss)

for i, wavpath in enumerate(wavpaths):
    wavpath = str(wavpath)
    filename = wavpath.split('/')[-1].split('.')[0]
    duration = librosa.get_duration(filename=wavpath)
    recording_id = re.search('^\d+', filename).group(0)

    # check whether annotations exist
    df_current = df_raw[(df_raw['recording id'] == int(recording_id)) & (df_raw['call type'] != '')]
    annotation_exists = len(df_current) > 0

    if annotation_exists:
        for item in df_current.to_dict('records'):
            pulse_duration = convert_time(item['duration'])
            annotation_duration = annotation_duration + pulse_duration

            call_type = item['call type'].strip()

            if call_type in call_map:
                call_map[call_type]['count'] += 1
                call_map[call_type]['duration'] += pulse_duration
            else:
                call_map[call_type] = {
                    'count': 1,
                    'duration': pulse_duration,
                    'id': len(call_map.keys()) + 1,
                }

    data.append([recording_id, duration, annotation_exists, item['ID']])

df = pd.DataFrame(data, columns=['recording_id', 'duration', 'annotation_exists', 'individual'])
df.to_csv('./analysis.csv')

print('How many individuals?')
print(len(df['individual'].drop_duplicates()))
print(df['individual'].drop_duplicates())

print('How about duration?')

print('annotation_duration:', annotation_duration)

print('Total duration of the annotated files')

print(df[df['annotation_exists'] == True]['duration'].sum())

print(df['duration'].describe())

pprint(call_map)