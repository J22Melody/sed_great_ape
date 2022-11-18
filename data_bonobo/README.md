# Data Bonobo

Assume you have the raw data here in `./raw`.

## Analysis

`python analysis.py`

There are 28 audio files by 7 individuals, annotated with call type information in the excel file.

Now have a look at the duration (in second) statistics:

```
Total duration of the annotated files: 677.0587291666666
annotation_duration: 61.83099999999994
```

Recording duration by files:

```
count     28.000000
mean      24.180669
std       32.163024
min        1.517521
25%        5.991672
50%        9.664167
75%       22.013411
max      110.891708
```

Call types:

```
{'NA': {'count': 2, 'duration': 0.32599999999999996, 'id': 13},
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
```

## Process

`python process.py`

Read the raw files and match every 0.02s segment to the annotations, as well as the wav2vec 2 features (size 768), save to `./wav2vec2`.

## Split

`python split.py`

Read the processed files and generate 3 different splits based on different random seeds.
