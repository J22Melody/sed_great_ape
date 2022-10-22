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

TXT_PATH = './raw/Klaus_Zuberbuhler_1994_2002/Raven Pro Tables'
WAV_PATH = './raw/Klaus_Zuberbuhler_1994_2002/Recordings'

data = []
annotation_duration = 0

wavpaths = Path(WAV_PATH).rglob('*.wav')

for i, wavpath in enumerate(wavpaths):
    wavpath = str(wavpath)
    filename = wavpath.split('/')[-1].split('.')[0]
    duration = librosa.get_duration(filename=wavpath)

    # check whether annotations exist
    txtpath = '{}/{}.txt'.format(TXT_PATH, filename)
    annotation_exist = os.path.exists(txtpath)

    if annotation_exist:
        # calculate duration
        with open(txtpath, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for item in reader:
                pulse_duration = float(item['End Time (s)']) - float(item['Begin Time (s)'])
                annotation_duration = annotation_duration + pulse_duration

    data.append([filename, duration, annotation_exist])

df = pd.DataFrame(data, columns=['filename', 'duration', 'annotation_exist'])
df.to_csv('./analysis.csv')

print('How many audio files? How many with anotations? How many have pulse level anotations?')

print(len(df))
print(len(df[df['annotation_exist'] == True]))

print('How about duration?')

print('Total duration of the annotated files')

print(df[df['annotation_exist'] == True]['duration'].sum())

print('annotation_duration:', annotation_duration)

print(df.describe())