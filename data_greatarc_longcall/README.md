# Data Greatarc Long Call

Assume you have the raw data here in `./raw`.

## Analysis

`python analysis.py`

There are 76 audio files, 66 of which have annotation files (can be used for binary classification), and 13 of which have pulse level annotation (can be used for multi-class classification).

Now have a look at the total duration (in second) and the total duration for each pulse level.

For the 66 annotated files:

```
Total duration of the annotated files: 4869.758911564626
annotation_duration: 1914.8147725079982
```

For the 13 pulse level annotated files:

```
Total duration of the pulse level annotated files: 1036.7140136054422

{'Bubble sub-pulse': {'count': 853, 'duration': 69.26793553600001, 'id': 5},
 'Full pulse': {'count': 445, 'duration': 382.98781547499993, 'id': 2},
 'Grumble sub-pulse': {'count': 23, 'duration': 1.9330904589999958, 'id': 6},
 'Pulse body': {'count': 176, 'duration': 106.47589200299993, 'id': 4},
 'Sub-pulse transitory element': {'count': 252,
                                  'duration': 20.980971026999967,
                                  'id': 3},
 'Unknown': {'count': 58, 'duration': 8.97822408899999, 'id': 1}}
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
