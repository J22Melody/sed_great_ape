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

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()

# The MPS backend is supported on MacOS 12.3+
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.has_mps else 'cpu'))
# device = torch.device('cpu')

DATA_PATH = './wav2vec2'
WAV_PATH = './raw/LC rhythm - all LC Tuanan'
ANNOTATION_PATH = './raw/ALL ELEMENTS Index.xlsx'

df_raw = pd.read_excel(ANNOTATION_PATH, keep_default_na=False)

# copy from ./analysis_new.log
pulse_level_map = {
    'Bubble sub-pulse': {'count': 4422, 'duration': 329.68749999999926, 'id': 5},
    'Full pulse': {'count': 1929, 'duration': 1762.548500000001, 'id': 2},
    'Grumble sub-pulse': {'count': 757, 'duration': 45.81939999999997, 'id': 6},
    'Grumph': {'count': 9, 'duration': 2.5993999999999957, 'id': 8},
    'Kiss-squeak': {'count': 9, 'duration': 2.6621999999999844, 'id': 7},
    'Pulse body': {'count': 816, 'duration': 553.7037000000003, 'id': 4},
    'Sub-pulse transitory element': {'count': 1068,
                                    'duration': 95.86570000000007,
                                    'id': 3},
    'Unknown': {'count': 0, 'duration': 0, 'id': 1}
}

def parse_raw():
    segment = 0.02
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)

    wavpaths = Path(WAV_PATH).rglob('*.wav')

    for i, wavpath in enumerate(wavpaths):
        wavpath = str(wavpath)
        filename = wavpath.split('/')[-1].split('.')[0]

        # remove 2 extremely long audio
        if filename == '13T8lcksqFugit' or filename == '86dat LC Sultan':
            continue

        # check whether annotations exist
        df_current = df_raw[(df_raw['Master File Name'] == filename) & (df_raw['Pulse level'] != '')]
        annotation_pulse_level = len(df_current) > 0

        if not annotation_pulse_level: 
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
            pulse_level = item['Pulse level'].strip() if item.get('Pulse level', '') else 'Unknown'

            # fix typo
            if pulse_level in ['Sub-pulse trnasitory element', 'Sub-pule transitory element', 'Sub-pulse transiotry element']:
                pulse_level = 'Sub-pulse transitory element'
            elif pulse_level in ['Bubble sub-pulseBubble sub-pulse', 'Buble sub-pulse']:
                pulse_level = 'Bubble sub-pulse'
            elif pulse_level in ['Pulse bpody', 'Puse body']:
                pulse_level = 'Pulse body'

            call = [float(item['Begin Time']), float(item['End Time']), pulse_level_map[pulse_level]['id']]
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