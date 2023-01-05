# Automatic Sound Event Detection and Classification of Great Ape Calls Using Neural Networks

## Data

Prepare each data set in a `data_xxx` directory, which contains:

- a `raw/` directory of raw data, including recordings and annotations
- some scripts to preprocess the raw data
- preprocessed data in possibly different splits

## Model

To train a model, make a model directory with a `config.json` inside of `models` directory.

### Train

For example:

```
python model.py -c models/chimp_wav2vec2_lstm_0/config.json
```

### TensorBoard

```
tensorboard --logdir=runs
```

### Results

See https://github.com/J22Melody/sed_great_ape/blob/main/model_stats.csv.

### Visualize

See https://github.com/J22Melody/sed_great_ape/tree/main/visualization.

## Legacy

See previous exploration logs in https://github.com/J22Melody/sed_great_ape/tree/main/legacy.
