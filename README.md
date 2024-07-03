# Automatic Sound Event Detection and Classification of Great Ape Calls Using Neural Networks

## Citation

Please cite [our preprint](https://arxiv.org/abs/2301.02214) (Arxiv) and [our paper](https://guarant.cz/icphs2023/508.pdf) (ICPhS) as follows:

```
@inproceedings{jiang-etal-2023-automatic,
    title = "Automatic Sound Event Detection and Classification of Great Ape Calls Using Neural Networks",
    author = {Jiang, Zifan and Soldati, Adrian and Schamberg, Isaac and Lameira, Adriano R and Moran, Steven},
    booktitle = "Proceedings of the 20th International Congress of Phonetic Sciences (ICPhS 2023)",
    pages = "3121--3125",
    month = august,
    year = "2023",
    address = "Prague, the Czech Republic"
}
```

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
