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
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()
# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
device = torch.device('cpu')

print(device)

RAW_PATH = './raw'
WAVEFORM_PATH = './waveform'
SPECTROGRAM_PATH = './spectrogram'
WAV2VEC2_PATH = './wav2vec2'

segment = 0.02
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)

wavpaths = Path(RAW_PATH).rglob('*.WAV')

df = pd.read_excel('{}/Pant hoots and phases_2.xlsx'.format(RAW_PATH), dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')
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

    # transform: resample to 16000 sample rate
    new_sample_rate = bundle.sample_rate
    transformed = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)

    # print(new_sample_rate)
    # print(transformed.size())

    # pad to multiply of 20ms
    length = transformed.size()[1]
    segment_length = int(segment * new_sample_rate)
    new_length = int(math.ceil(length / segment_length) * segment_length)
    target = torch.zeros(1, new_length).to(device)
    target[:, :length] = transformed
    target_wav_vec2 = torch.zeros(1, new_length + segment_length).to(device) # HACK: wav2vec2 produces length - 1
    target_wav_vec2[:, :length] = transformed
    print(target.size())

    # waveform
    waveform_features = target.view(-1, segment_length)
    print(waveform_features.size())

    # spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(
        hop_length=segment_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    spectrogram_features = spectrogram(target).squeeze().transpose(0, 1)[:-1, :]
    print(spectrogram_features.size())

    # wav2vec2
    with torch.inference_mode():
        features, _ = model.extract_features(target_wav_vec2)
    wav2vec2_features = features[len(features) - 1]
    wav2vec2_features = wav2vec2_features.squeeze().cpu()
    print(wav2vec2_features.size())

    # read annotations
    row = df[df['File Name'] == filename]

    calls = [
        [float(row['Intro starts (s)']), float(row['Intro ends (s)']), 1],
        [float(row['Build-up starts']), float(row['Build-up ends']), 2],
        [float(row['Climax starts']), float(row['Climax ends']), 3],
        [float(row['Let-down starts']), float(row['Let-down ends']), 4],
    ]

    # create y
    y = [0] * wav2vec2_features.size()[0]
    for begin, end, call_type in calls:
        begin_index = math.floor(begin / segment)
        end_index = math.ceil(end / segment)
        for i in range(begin_index, end_index):
            if i < len(y):
                y[i] = call_type
    y = torch.tensor(y).view(-1, 1)

    # write files
    Path(WAVEFORM_PATH).mkdir(parents=True, exist_ok=True)
    data = torch.cat((y, waveform_features), dim=1)
    np.savetxt('{}/{}.csv'.format(WAVEFORM_PATH, filename), data.numpy(), delimiter=',')

    Path(SPECTROGRAM_PATH).mkdir(parents=True, exist_ok=True)
    data = torch.cat((y, spectrogram_features), dim=1)
    np.savetxt('{}/{}.csv'.format(SPECTROGRAM_PATH, filename), data.numpy(), delimiter=',')

    Path(WAV2VEC2_PATH).mkdir(parents=True, exist_ok=True)
    data = torch.cat((y, wav2vec2_features), dim=1)
    np.savetxt('{}/{}.csv'.format(WAV2VEC2_PATH, filename), data.numpy(), delimiter=',')

print("--- %s seconds ---" % (time.time() - start_time))



