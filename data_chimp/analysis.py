import math
import pandas as pd
import numpy as np
import librosa

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# DATA ANALYSIS

df = pd.read_excel('./raw/Pant hoots and phases_2.xlsx', dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')

# filter
df = df[df['For Steven'] == 1.0]

print('How many individuals?')
print(len(df['ID'].drop_duplicates()))
print(df['ID'].drop_duplicates())

# how many units?
unit_count = 0
for row in df.to_dict('records'):
    if not math.isnan(row['Intro starts (s)']):
        unit_count += 1
    if not math.isnan(row['Build-up starts']):
        unit_count += 1
    if not math.isnan(row['Climax starts']):
        unit_count += 1
    if not math.isnan(row['Let-down starts']):
        unit_count += 1

print('how many units?', unit_count)

# get duration and check file existence
durations = []
for filename in list(df['File Name']):
    duration = librosa.get_duration(filename='./raw/{}.wav'.format(filename.replace(';', '_')))
    durations.append(duration)

df['duration'] = durations
df = df.replace(np.nan, 0)

df['intro_duration'] = df['Intro ends (s)'] - df['Intro starts (s)']
df['build_up_duration'] = df['Build-up ends'] - df['Build-up starts']
df['climax_duration'] = df['Climax ends'] - df['Climax starts']
df['let_down_duration'] = df['Let-down ends'] - df['Let-down starts']

print(df)

df_sum = df.sum(numeric_only=True)
print(df_sum)

print(df['duration'].describe())

