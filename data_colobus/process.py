import csv
import math
import random
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()

# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
# However, the generated features strangely lead to poor training performamce, fallback to cpu
device = torch.device('cpu')

TXT_PATH = './raw/Klaus_Zuberbuhler_1994_2002/Raven Pro Tables'
WAV_PATH = './raw/Klaus_Zuberbuhler_1994_2002/Recordings'
DATA_PATH = './wav2vec2'

def parse_raw():
    segment = 0.02
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)

    wavpaths = Path(WAV_PATH).rglob('*.wav')

    for i, wavpath in enumerate(wavpaths):
        wavpath = str(wavpath)
        filename = wavpath.split('/')[-1].split('.')[0]
        duration = librosa.get_duration(filename=wavpath)

        # remove extremely long audio
        if duration > 200:
            continue

        # check whether annotations exist
        txtpath = '{}/{}.txt'.format(TXT_PATH, filename)
        annotation_exist = os.path.exists(txtpath)
        if not annotation_exist: 
            continue

        print('{}: processing {} ...'.format(i, wavpath))

        df = pd.read_csv(txtpath, sep='\t')

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

        # pad to multiply of 20ms
        length = transformed.size()[1]
        segment_length = int(segment * new_sample_rate)
        new_length = int(math.ceil(length / segment_length) * segment_length)
        target = torch.zeros(1, new_length + segment_length).to(device) # HACK: wav2vec2 produces length - 1
        target[:, :length] = transformed
        # print(target.size())

        # wav2vec2
        with torch.inference_mode():
            features, _ = model.extract_features(target)
        output_features = features[len(features) - 1]
        output_features = output_features.squeeze().cpu()
        print(output_features.size())

        # read annotations
        calls = []
        for item in df.to_dict('records'):
            call = [float(item['Begin Time (s)']), float(item['End Time (s)']), 1]
            calls.append(call)

        # create y
        y = [0] * output_features.size()[0]
        for begin, end, call_type in calls:
            begin_index = math.floor(begin / segment)
            end_index = math.ceil(end / segment)
            for i in range(begin_index, end_index):
                y[i] = call_type
        y = torch.tensor(y).view(-1, 1)

        data = torch.cat((y, output_features), dim=1)

        Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
        np.savetxt('{}/{}.csv'.format(DATA_PATH, filename), data.numpy(), delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

parse_raw()