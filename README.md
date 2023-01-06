# Automatic Sound Event Detection and Classification of Great Ape Calls Using Neural Networks

## Citation

Please cite [our paper](https://arxiv.org/abs/2301.02214).

<!-- as follows:

```
@article{jiang2022machine,
  title={Machine Translation between Spoken Languages and Signed Languages Represented in SignWriting},
  author={Jiang, Zifan and Moryossef, Amit and M{\"u}ller, Mathias and Ebling, Sarah},
  journal={arXiv preprint arXiv:2210.05404},
  year={2022}
}
``` -->

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
