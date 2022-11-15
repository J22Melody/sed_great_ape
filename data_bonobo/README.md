# Data Bonobo

Assume you have the raw data here in `./raw`.

## Analysis

`python analysis.py`

There are 19 audio files by 4 individuals, annotated with call type information in the excel file.

Now have a look at the duration (in second) statistics:

```
Total duration of the annotated files: 486.8056041666667
annotation_duration: 42.456999999999994
```

Recording duration by files:

```
count     19.000000
mean      25.621348
std       33.824196
min        1.517521
25%        6.798448
50%        8.838646
75%       24.597323
max      110.891708
```

Call types:

```
{'ba': {'count': 2, 'duration': 0.569, 'id': 13},
 'gr': {'count': 24, 'duration': 2.863, 'id': 4},
 'hh': {'count': 62, 'duration': 13.508999999999997, 'id': 8},
 'hhsb': {'count': 1, 'duration': 0.506, 'id': 9},
 'in': {'count': 12, 'duration': 0.9899999999999999, 'id': 15},
 'lh': {'count': 2, 'duration': 0.956, 'id': 16},
 'pe': {'count': 6, 'duration': 0.7, 'id': 11},
 'pg': {'count': 19, 'duration': 1.385, 'id': 3},
 'py': {'count': 32, 'duration': 4.433, 'id': 2},
 'sb': {'count': 33, 'duration': 9.000000000000002, 'id': 5},
 'sc': {'count': 1, 'duration': 0.127, 'id': 14},
 'scb': {'count': 10, 'duration': 2.84, 'id': 12},
 'wb': {'count': 10, 'duration': 2.265, 'id': 10},
 'wh': {'count': 1, 'duration': 0.622, 'id': 7},
 'wi': {'count': 1, 'duration': 0.394, 'id': 1},
 'ye': {'count': 10, 'duration': 1.298, 'id': 6}}
```

## Process

`python process.py`

Read the raw files and match every 0.02s segment to the annotations, as well as the wav2vec 2 features (size 768), save to `./wav2vec2`.

## Split

`python split.py`

Read the processed files and generate 3 different splits based on different random seeds.
