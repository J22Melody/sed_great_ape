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

TXT_PATH = './raw/LC rhythm - all LC Tuanan/Raven tables'
WAV_PATH = './raw/LC rhythm - all LC Tuanan'

data = []
pulse_level_map = {'Unknown': {'id': 1, 'count': 0, 'duration': 0}}
annotation_duration = 0

wavpaths = Path(WAV_PATH).rglob('*.wav')

for i, wavpath in enumerate(wavpaths):
    wavpath = str(wavpath)
    filename = wavpath.split('/')[-1].split('.')[0]
    duration = librosa.get_duration(filename=wavpath)

    # check whether annotations exist
    txtpath = '{}/{}.Table.1.selections.txt'.format(TXT_PATH, filename)
    annotation_exist = os.path.exists(txtpath)
    annotation_pulse_level = False

    if annotation_exist:
        # calculate duration
        with open(txtpath, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            annotation_pulse_level = any([item.get('Pulse level', '') for item in reader])

            with open(txtpath, newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')

                for item in reader:
                    pulse_duration = float(item['End Time (s)']) - float(item['Begin Time (s)'])
                    annotation_duration = annotation_duration + pulse_duration

                    if annotation_pulse_level:
                        pulse_level = item['Pulse level'].strip() if item.get('Pulse level', '') else 'Unknown'

                        # fix typo
                        if pulse_level in ['Sub-pulse trnasitory element', 'Sub-pule transitory element', 'Sub-pulse transiotry element']:
                            pulse_level = 'Sub-pulse transitory element'

                        if pulse_level in pulse_level_map:
                            pulse_level_map[pulse_level]['count'] += 1
                            pulse_level_map[pulse_level]['duration'] += pulse_duration
                        else:
                            pulse_level_map[pulse_level] = {
                                'count': 1,
                                'duration': pulse_duration,
                                'id': len(pulse_level_map.keys()) + 1,
                            }

    data.append([filename, duration, annotation_exist, annotation_pulse_level])

df = pd.DataFrame(data, columns=['filename', 'duration', 'annotation_exist', 'annotation_pulse_level'])
df.to_csv('./analysis.csv')

print('How many audio files? How many with anotations? How many have pulse level anotations?')

print(len(df))
print(len(df[df['annotation_exist'] == True]))
print(len(df[df['annotation_pulse_level'] == True]))

print('How about duration?')

print('Total duration of the annotated files')

print(df[df['annotation_exist'] == True]['duration'].sum())

print('annotation_duration:', annotation_duration)

print('Total duration of the pulse level annotated files')

print(df[df['annotation_pulse_level'] == True]['duration'].sum())

pprint(pulse_level_map)