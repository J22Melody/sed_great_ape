# Data Chimp

Assume you have the raw data here in `./raw`.

## Analysis

`python analysis.py`

There are 235 audio files, annotated with phase information in the excel file.

Now have a look at the total duration (in second) and the total duration for each phase:

```
duration               1964.217937
intro_duration          976.710000
build_up_duration       427.426000
climax_duration         395.924000
let_down_duration       155.185000
```

## Process

`python process.py`

Read the raw files and match every 0.02s segment to the annotations, as well as the following features:

- `./waveform` (size 320)
- `./spectrogram` (size 201)
- `./wav2vec2` (size 768)

## Split

`python split.py`

Read the processed files and generate 3 different splits based on different random seeds.
