import pandas as pd
import numpy as np
import librosa

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# DATA ANALYSIS

df = pd.read_excel('./data_chimp/raw/Pant hoots and phases_2.xlsx', dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')

# filter
df = df[df['For Steven'] == 1.0]

# get duration and check file existence
durations = []
for filename in list(df['File Name']):
    duration = librosa.get_duration(filename='./data_chimp/raw/{}.wav'.format(filename.replace(';', '_')))
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

