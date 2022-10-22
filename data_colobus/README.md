# Data Olive Colobus

Assume you have the raw data (https://github.com/QuentinGallot/OliveColobusDatabase) here in `./raw`.

## Analysis

`python analysis.py`

There are 649 audio files, 284 of which have annotation files (can be used for binary classification).

Now have a look at the duration (in second):

```
Total duration of the annotated files: 53494.50657596372
annotation_duration: 888.9374572619978

         duration
count  649.000000
mean   187.006206
std     14.497718
min    120.151633
25%    181.217234
50%    186.276281
75%    191.222132
max    284.847891
```

Very unbalanced! We might need to adjust the class weight for efficient training.

Note the audio files have ~180s duration each, we remove too long (> 200s) audio files at training (266 files remain).

## Process

`python process.py`

Read the raw files and match every 0.02s segment to the annotations, as well as the wav2vec 2 features (size 768), save to `./wav2vec2`.

## Split

`python split.py`

Read the processed files and generate 3 different splits based on different random seeds.
