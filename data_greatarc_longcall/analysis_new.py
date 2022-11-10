from __future__ import annotations
import pandas as pd
import numpy as np
import librosa
import os
import csv
from pathlib import Path
from pprint import pprint


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

WAV_PATH = './raw/LC rhythm - all LC Tuanan'
ANNOTATION_PATH = './raw/ALL ELEMENTS Index.xlsx'

df_raw = pd.read_excel(ANNOTATION_PATH, keep_default_na=False)

data = []
pulse_level_map = {'Unknown': {'id': 1, 'count': 0, 'duration': 0}}
annotation_duration = 0

wavpaths = Path(WAV_PATH).rglob('*.wav')

for i, wavpath in enumerate(wavpaths):
    wavpath = str(wavpath)
    filename = wavpath.split('/')[-1].split('.')[0]
    duration = librosa.get_duration(filename=wavpath)

    # check whether annotations exist
    df_current = df_raw[(df_raw['Master File Name'] == filename) & (df_raw['Pulse level'] != '')]
    annotation_pulse_level = len(df_current) > 0

    if annotation_pulse_level:
        for item in df_current.to_dict('records'):
            pulse_duration = float(item['End Time']) - float(item['Begin Time'])
            annotation_duration = annotation_duration + pulse_duration

            pulse_level = item['Pulse level'].strip() if item.get('Pulse level', '') else 'Unknown'

            # fix typo
            if pulse_level in ['Sub-pulse trnasitory element', 'Sub-pule transitory element', 'Sub-pulse transiotry element']:
                pulse_level = 'Sub-pulse transitory element'
            elif pulse_level in ['Bubble sub-pulseBubble sub-pulse', 'Buble sub-pulse']:
                pulse_level = 'Bubble sub-pulse'
            elif pulse_level in ['Pulse bpody', 'Puse body']:
                pulse_level = 'Pulse body'

            if pulse_level in pulse_level_map:
                pulse_level_map[pulse_level]['count'] += 1
                pulse_level_map[pulse_level]['duration'] += pulse_duration
            else:
                pulse_level_map[pulse_level] = {
                    'count': 1,
                    'duration': pulse_duration,
                    'id': len(pulse_level_map.keys()) + 1,
                }

    data.append([filename, duration, annotation_pulse_level, item['Individual']])

df = pd.DataFrame(data, columns=['filename', 'duration', 'annotation_pulse_level', 'individual'])
df.to_csv('./analysis_new.csv')

print('How many individuals?')
print(len(df['individual'].drop_duplicates()))
print(df['individual'].drop_duplicates())

print('How many audio files? How many have pulse level anotations?')

print(len(df))
print(len(df[df['annotation_pulse_level'] == True]))

print('How about duration?')

print('annotation_duration:', annotation_duration)

print('Total duration of the pulse level annotated files')

print(df[df['annotation_pulse_level'] == True]['duration'].sum())

pprint(pulse_level_map)