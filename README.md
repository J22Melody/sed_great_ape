# Audio Exploration

## Legacy

See previous exploration logs in https://github.com/bambooforest/audio_exploration/tree/main/legacy.

## Data

Prepare each data set in a `data_xxx` directory, which contains:

- a `raw/` directory of raw data
- some scripts to preprocess the raw data
- preprocessed data in possibly different splits

## Model

To train a model, make a model directory with a `config.json` inside of `models` directory.

### Train

```
python model.py -c models/chimp_lstm_8/config.json
```

### Results

See https://github.com/bambooforest/audio_exploration/blob/main/results/metrics.csv.

### Visualize

See https://github.com/bambooforest/audio_exploration/tree/main/visualization.

## To Reminds @J22Melody of What We Are Working with ...

![great apes](https://github.com/bambooforest/audio_exploration/blob/4a45ce937df5ac3dc592d3c116250fcc6ce864ef/great_apes.png?raw=true)
