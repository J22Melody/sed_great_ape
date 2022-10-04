# Audio Exploration

## Legacy

See previous exploration logs in 

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

### Visualize

See 