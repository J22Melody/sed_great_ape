# Data Greatarc Long Call

Assume you have the raw data (https://www.dropbox.com/home/GreatARC/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051) here in `./raw`.

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

## Analysis (New)

The old analysis above based on the raven tables do not include complete annotations, the new analysis is based on the .xlsx annotation file.

`python analysis_new.py`

There are 76 audio files, 65 of which have pulse level annotation files (can be used for multi-class classification).

Now have a look at the total duration (in second) and the total duration for each pulse level.

For the 65 annotated files:

```
{'Bubble sub-pulse': {'count': 4422, 'duration': 329.68749999999926, 'id': 5},
 'Full pulse': {'count': 1929, 'duration': 1762.548500000001, 'id': 2},
 'Grumble sub-pulse': {'count': 757, 'duration': 45.81939999999997, 'id': 6},
 'Grumph': {'count': 9, 'duration': 2.5993999999999957, 'id': 8},
 'Kiss-squeak': {'count': 9, 'duration': 2.6621999999999844, 'id': 7},
 'Pulse body': {'count': 816, 'duration': 553.7037000000003, 'id': 4},
 'Sub-pulse transitory element': {'count': 1068,
                                  'duration': 95.86570000000007,
                                  'id': 3},
 'Unknown': {'count': 0, 'duration': 0, 'id': 1}}
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
