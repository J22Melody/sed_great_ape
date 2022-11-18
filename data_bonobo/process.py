import csv
import math
import random
import time
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()

# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
# However, the generated features strangely lead to poor training performamce, fallback to cpu
device = torch.device('cpu')

DATA_PATH = './wav2vec2'
WAV_PATH = './raw/bonobo calls'
ANNOTATION_PATH = './raw/metadata.csv'

df_raw = pd.read_csv(ANNOTATION_PATH, keep_default_na=False)

# copy from ./analysis.log
call_map = {'NA': {'count': 2, 'duration': 0.32599999999999996, 'id': 13},
 'ba': {'count': 2, 'duration': 0.569, 'id': 17},
 'gr': {'count': 24, 'duration': 2.863, 'id': 4},
 'hh': {'count': 93, 'duration': 22.822, 'id': 7},
 'hhsb': {'count': 1, 'duration': 0.506, 'id': 9},
 'hhsc': {'count': 3, 'duration': 1.7719999999999998, 'id': 15},
 'in': {'count': 34, 'duration': 2.5500000000000003, 'id': 12},
 'lh': {'count': 35, 'duration': 4.07, 'id': 11},
 'pe': {'count': 8, 'duration': 0.873, 'id': 10},
 'pg': {'count': 19, 'duration': 1.385, 'id': 3},
 'py': {'count': 37, 'duration': 5.06, 'id': 2},
 'sb': {'count': 34, 'duration': 9.122000000000002, 'id': 5},
 'sc': {'count': 1, 'duration': 0.127, 'id': 18},
 'scb': {'count': 10, 'duration': 2.84, 'id': 16},
 'wb': {'count': 10, 'duration': 2.265, 'id': 14},
 'wh': {'count': 1, 'duration': 0.622, 'id': 8},
 'wi': {'count': 6, 'duration': 2.761, 'id': 1},
 'ye': {'count': 10, 'duration': 1.298, 'id': 6}}

def convert_time(input):
    mm, ss = input.split(':')
    return int(mm) * 60 + float(ss)

def parse_raw():
    segment = 0.02
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)

    wavpaths = Path(WAV_PATH).rglob('*.wav')

    for i, wavpath in enumerate(wavpaths):
        wavpath = str(wavpath)
        filename = wavpath.split('/')[-1].split('.')[0]
        recording_id = re.search('^\d+', filename).group(0)

        # check whether annotations exist
        df_current = df_raw[(df_raw['recording id'] == int(recording_id)) & (df_raw['call type'] != '')]
        annotation_exists = len(df_current) > 0

        if not annotation_exists:
            continue

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
        for item in df_current.to_dict('records'):
            call_type = item['call type'].strip()
            start_time = convert_time(item['start time'])
            duration = convert_time(item['duration'])

            call = [start_time, start_time + duration, call_map[call_type]['id']]
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
        np.savetxt('{}/{}.csv'.format(DATA_PATH, recording_id), data.numpy(), delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

parse_raw()