import csv
import math
import random
import itertools
import time
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchaudio

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()
# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
device = torch.device('cpu')

print(device)

WAV_PATH = './data_chimp/raw'
DATA_PATH = './data_chimp/wav2vec2'

segment = 0.02
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)

wavpaths = Path(WAV_PATH).rglob('*.WAV')

df = pd.read_excel('{}/Pant hoots and phases_2.xlsx'.format(WAV_PATH), dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')
df = df.replace(np.nan, 0)
df['File Name'] = df['File Name'].str.replace(';', '_')

for i, wavpath in enumerate(wavpaths):
    wavpath = str(wavpath)
    filename = wavpath.split('/')[-1].replace('.WAV', '')

    print('{}: processing {} ...'.format(i, wavpath))

    # read wav file
    waveform, sample_rate = torchaudio.load(wavpath)
    waveform = waveform.to(device)

    # print(sample_rate)
    # print(waveform.size())

    # transform
    new_sample_rate = bundle.sample_rate
    transformed = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)

    # print(new_sample_rate)
    # print(transformed.size())

    # padding to full seconds + 20ms (HACK for wav2vec2 length minus 1)
    # length = transformed.size()[1]
    # new_length = int((math.ceil(length / new_sample_rate) + segment) * new_sample_rate)
    # target = torch.zeros(1, new_length).to(device)
    # target[:, :length] = transformed

    target = transformed

    # print(target.size())

    # wav2vec2
    with torch.inference_mode():
        features, _ = model.extract_features(target)
    output_features = features[len(features) - 1]
    output_features = output_features.squeeze().cpu()
    print(output_features.size())

    row = df[df['File Name'] == filename]

    # read annotations
    calls = [
        [float(row['Intro starts (s)']), float(row['Intro ends (s)']), 1],
        [float(row['Build-up starts']), float(row['Build-up ends']), 2],
        [float(row['Climax starts']), float(row['Climax ends']), 3],
        [float(row['Let-down starts']), float(row['Let-down ends']), 4],
    ]

    # create y
    y = [0] * output_features.size()[0]
    for begin, end, call_type in calls:
        begin_index = math.floor(begin / segment)
        end_index = math.ceil(end / segment)
        for i in range(begin_index, end_index):
            if i < len(y):
                y[i] = call_type
    y = torch.tensor(y).view(-1, 1)

    data = torch.cat((y, output_features), dim=1)

    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    np.savetxt('{}/{}.csv'.format(DATA_PATH, filename), data.numpy(), delimiter=',')

print("--- %s seconds ---" % (time.time() - start_time))



