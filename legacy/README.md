# audio_exploration

## [22.07.2022] Transformer

Why?

- Transformer has outperformed and replaced LSTM in many NLP tasks, e.g., machine translation.
- Transformer is a more generalized encoder architecture, with less inductive bias on input data (i.e., RNN for time series input and CNN for image input)
  - meaning it could work worse with constrained data conditions, but has a better ability to generalize, especially on large data conditions.

So I did this experiment on replacing the LSTM with a Transformer model, but I did not expect it could outperform much more than the LSTM in our low data condition.

The results are close to LSTM:

```
Test Epoch: accuracy: 0.68 
 precision: [0.9205298  0.         0.47841727 0.12380952 0.82397004 0.05577689
 0.        ] 
 recall: [0.81925344 1.         0.63942308 0.61904762 0.35369775 1.
 1.        ] 
 f1: [0.86694387 0.         0.5473251  0.20634921 0.49493813 0.10566038
 0.        ] 
 f1_avg: 0.7173748556916042
```

But seems to be more robust when generalizing from training to test and dev:

![image](https://user-images.githubusercontent.com/2316987/180486554-59893cb3-c19f-4454-87a0-faf7e345d726.png)

I think we could write results and do further experiments on new data on both models.

## [03.07.2022] Fix Bug, Rerun Experiment, and Visualization

### Fixed Test Performance 

I spot a very hidden but major bug in the model training code, which, in a word, led to training/test dataset overlapping in previous experiments. So I fix the bug and rerun the experiment (on the 13 files with pulse level annotation), and it does affect the model performance. 

https://github.com/bambooforest/audio_exploration/blob/visualization/longcall/rnn.log

```
Test Epoch: accuracy: 0.70 
 precision: [0.96845426 0.         0.456942   0.0862069  0.86376022 0.03100775
 0.        ] 
 recall: [0.80419122 1.         0.625      0.23809524 0.5096463  0.28571429
 1.        ] 
 f1: [0.87871199 0.         0.52791878 0.12658228 0.64105157 0.05594406
 0.        ] 
 f1_avg: 0.7552242056919353
```

<img width="769" alt="image" src="https://user-images.githubusercontent.com/2316987/178314751-a4abeefd-2f1a-44b2-bae3-e8668c4f2829.png">

<img width="1379" alt="image" src="https://user-images.githubusercontent.com/2316987/178314954-b9508b7e-c97e-4a77-838d-e80c7d69164e.png">

Now it is not as good as before, but at least it can still detect calls quite well (0.88 f1 for class 0), and the rest classes' performance depends highly on the number of samples of each class.

### Visualization

I create a Web app to visualize the model's prediction:

<img width="1917" alt="image" src="https://user-images.githubusercontent.com/2316987/178315061-037ace00-0647-4270-a51b-b633f8ee879d.png">

To run a built version locally, navigate into the directory `audio_exploration/longcall/visualization/build`, and run `python3 -m http.server`. It starts a local server where you can see the visualization.

## [29.03.2022] Recurrent Sequence Modeling on Long Calls

Following the exploration in https://github.com/bambooforest/audio_exploration/pull/3, I experimented with a more sophisticated model architecture.  I modeled each whole long call audio (~60 s) as a long sequence in a LSTM model, then output a class distribution of pulse levels at each time step (20 ms segment). It turns out to be a big success, the average f1 score on test set reaches 0.91.

### Inspiration

Sequence modeling are very common in NLP, for example, we build a recurrent language model to model a sentence (a sequence of word), then predict the part of speech tag of each word at each time step. In audio field, I find a similar approach in https://distill.pub/2017/ctc/, where they tackle the problem of speech recognition by a CTC loss (because the audio and the target sentence are not aligned). 

In our case, fortunately, the audio and the annotation of pulse levels are strictly aligned by time, so we can simply use a recurrent model and a cross-entropy loss to handle it.

### Data

There are 66 annotated long call audio files from Greatarc dataset, 13 of them have pulse level annotation. So we can naturally train multi-class models on the 13 files, or we can train binary classifiers (simply whether there is a call or not) on all the 66 files.

I decide to go the multi-class way, using an 8/1/1 data split, I have the file `4T10lcFugit` as test set, the file `5T10lcFugit` as dev set, and the rest 11 files as training set. I also did a contrast experiment on 66 files, by giving all the annotated entries without pulse level a class `unknown`.

All the experiments happened inside `./longcall`, so to preprocess the data, first enter the directory, then run `python data_wav2vec2.py` - read the raw data, extract `wav2vec2` features for each 20ms segment (introduced in https://github.com/bambooforest/audio_exploration/pull/2, approach 3) and align them with annotations. 

The result is in https://github.com/bambooforest/audio_exploration/blob/rnn_clf/longcall/data_wav2vec2.log, most importantly, the statistics of class numbers before and after segmentation:

`{'Unknown': {'count': 2872, 'id': 1}, 'Full pulse': {'count': 445, 'id': 2}, 'Sub-pulse transitory element': {'count': 252, 'id': 3}, 'Pulse body': {'count': 176, 'id': 4}, 'Bubble sub-pulse': {'count': 853, 'id': 5}, 'Grumble sub-pulse': {'count': 23, 'id': 6}}`

`{0.0: 164462, 1.0: 60907, 2.0: 9163, 3.0: 923, 4.0: 5458, 5.0: 4019, 6.0: 118}`

Again, `0` represents segment without calls, and there are many unknown samples since most of the files do not have pulse level information, in the 13 files that have pulse level, we only have 323 unknown samples.

### Training and Results

Run `python rnn.py`, the experiments finish within one hour on a GPU. The evaluation results (https://github.com/bambooforest/audio_exploration/blob/rnn_clf/longcall/rnn.log) on test set (for each class):

```
Test Epoch: accuracy: 0.87 
 precision: [0.99128129 0.07217281 0.93826027 0.9189463  0.97115908 0.91433147
 0.06640405] 
 recall: [0.84752764 0.87925697 0.89228419 0.98266522 0.9809454  0.85245086
 1.        ] 
 f1: [0.91378531 0.13339596 0.91469486 0.94973822 0.97602771 0.88230749
 0.12453826] 
 f1_avg: 0.9120521962326353
```

To get a more direct feeling of the predictions, see in parallel https://github.com/bambooforest/audio_exploration/blob/rnn_clf/longcall/rnn.pred.txt and https://github.com/bambooforest/audio_exploration/blob/rnn_clf/longcall/rnn.target.txt. 

The statistics, in general, are very satisfying, most of the known classes perform very well (over 0.90), the `unknown` class perform badly as expected, the last class `Grumble sub-pulse` perform badly probably because of lack of samples.

To dig more into the training process, I use `tensorboard` to see the graphs (`tensorboard` files in `./runs`), here are some:

<img width="703" alt="image" src="https://user-images.githubusercontent.com/2316987/160625783-64c5a67f-0ac3-4183-be0a-5efe5b11963f.png">
<img width="1117" alt="image" src="https://user-images.githubusercontent.com/2316987/160625895-d4632cd1-d8b6-4807-b34d-91a28ded89df.png">

(orange lines represent experiment on 13 files without too many `unknown` class samples, blue lines represent experiment on 66 files with many `unknown` class samples)

All these plots also show clearly that the `unknown` class samples do confuse the model to correctly classify, however, we can also expect if we annotate pulse level for all the 66 files, the model would be much more robust.

### Conclusion

I think the results are very good to me, so I would probably stop my audio exploration at the moment on these data. We might try to use the models to work on new real data, and do some human evaluation to see how well it works indeed ;)

Some takeaway on my side:

- We can successfully train classifiers to extract monkey calls from audio, no matter binary or multi-class.
- One key aspect is data (as always). We are still in a very low-resource setting, but as long as we have balanced, well-structured data, it's promising to train a good classifier.
- Another thing is how we want to generalize. In https://github.com/bambooforest/audio_exploration/pull/2 I find that data cannot generalize across files in the `OliveColobusDatabase ` dataset (but do not know why). In https://github.com/bambooforest/audio_exploration/pull/3, I find that the reason might be: different individuals? Different call repertoires, etc. It is definitely hard to classify when, say, we train a model on some data, then there comes a new individual with totally unseen repertoires. Conversely, if we, say, have (at least) tens of audio of Kelly, which contains a comprehensive list of his repertoires, then we can confidently deal with a future file of his (at least) (semi-)automatically.
- In terms of features and model architecture, I think especially the features, all kinds of feature selection would work to some extent, raw wavforms, spectrogram, wav2vec, etc. It is similar in terms of modeling, nevertheless, the model architecture in this couple of experiments turns out to be the most sophisticated and performs the best.

## [17.03.2022] Same Experiments on Greatarc Dataset

I tried to rerun some of the same experiments on the Greatarc data, as I performed in https://github.com/bambooforest/audio_exploration/pull/2. The experiments are rather rough, but a main finding is that the Greatarc data does generate across files, as opposed to the OliveColobusDatabase data. I think this is a very good signal to do further exploration.

### Binary Classification on Calls

Setup:

- Binary classifies if a 1-second segment is a call
- Data comes from `./data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301`
- Scripts and logs are in `./greatarc`

Observation on Data: 
- There are not many audio files, but all files are very long (~20 minutes), not all of them have many annotations.
- There are not many individuals, but the files for Kelly and YetYeni have good annotations in terms of training a classifier.
- Still a very class-imbalanced task since most of the 1-second segments are not calls.

Experiments and results:
- Learn from `Kelly experiment Tiger 19-12-2010` to predict `Kelly experiment spots 11-1-2011`: AUC-PR (area under the precision-recall curve, the higher the better) = 0.72 (https://github.com/bambooforest/audio_exploration/blob/greatarc/greatarc/greatarc_clf_kelly_19_to_11_test.png)
- Learn from `Kelly experiment Tiger 19-12-2010` to predict `YetYeni experiment tiger sheet 10-02-2011 (1)_1st half`: AUC-PR = 0.41 (https://github.com/bambooforest/audio_exploration/blob/greatarc/greatarc/greatarc_clf_kelly_to_yetyeni_test.png)
- Learn from `YetYeni experiment tiger sheet 10-02-2011 (1)_1st half` to predict `Kelly experiment spots 11-1-2011`: AUC-PR = 0.64 (https://github.com/bambooforest/audio_exploration/blob/greatarc/greatarc/greatarc_clf_yetyeni_to_kelly_11_test.png)

Interpretations:
- We can successfully learn and predict the calls of the same individual (Kelly).
- When doing so across different individuals, this might also work well (from YetYeni to Kelly), this might not work so well (from Kelly to YetYeni).
- Why? Rather than attributing to the difference between individuals, after checking the annotation files, I believe the main cause is that from the data YetYeni has call types that Kelly does not have, such as `Complex call` and `Biarticulated voiced element`, while Kelly only has `Grumph`, `Kiss-squeak` and `Rolling call`. 
- So I would conclude that it is promising to train an effective classifier to find out calls from audio files, given a training set that contains all the call types we are interested in, ideally from more than one individuals, to prevent overfitting to paticular individual.

### Multi-class Classification on Calls

Setup:

- Extend previous experiments to multi-class, classifies if a 1-second segment is a non-call, `Grumph`, `Kiss-squeak`, `Rolling call`, or an unknown/other call.
- Data comes from `./data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301`
- Scripts and logs are in `./greatarc`

Observation on Data:

- The problem of imbalanced classes is more severe: the most common call types are `Grumph`, `Kiss-squeak` and `Rolling call`, yet in training set there are 6843 non-call segments, 438 `Kiss-squeak`s, 59 `Grumph`s and 41 `Rolling call`s.

Experiments and results:
- Train a classifier to tell different call types? f1 score for the 3 classes: 0.84, 0.22222222, 0.46666667 (https://github.com/bambooforest/audio_exploration/blob/greatarc/greatarc/cnn_nonzero.log)
- Train a classifier to tell different call types and non-calls? f1 score for the 4 classes: 0.0869565, 0.0, 0.10526316, 0.96308725 (https://github.com/bambooforest/audio_exploration/blob/greatarc/greatarc_log/cnn.log)

Interpretations:
- The imbalanced distribution of classes is tricky to deal with, and the minority samples are just too few. 

### Multi-class Classification on Long Calls

Setup:

- Multi-class classifies if a 1-second segment is a non-call, or a type of pulse from long calls
- Data comes from `./data_greatarc/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051`
- Scripts and logs are in `./longcall`

Observation on Data:

- The long call data is much more friendly for training a classifier! More audio files (each with a short duration of ~60s) and annotations.
- The class distribution on long call data is more even (yet there are many calls unannotated so they are with unknown pulse type):

```
{'Unknown': {'count': 2872, 'id': 1}, 'Full pulse': {'count': 445, 'id': 2}, 'Sub-pulse transitory element': {'count': 252, 'id': 3}, 'Pulse body': {'count': 176, 'id': 4}, 'Bubble sub-pulse': {'count': 853, 'id': 5}, 'Grumble sub-pulse': {'count': 23, 'id': 6}}
```
- But many pulse types are very short, so after 1-second segmentation, the distribution of classes becomes (0 stands for non-calls):

```
Labels in train:  {0.0: 1545, 1.0: 1748, 2.0: 506, 3.0: 46, 4.0: 2, 5.0: 13}
```

Experiments and results:
- Learn from 80% of the files and predict the rest: f1 score of 5 classes: 0.73981191, 0.71527778, 0.34693878, 0.66666667 0.66666667 (https://github.com/bambooforest/audio_exploration/blob/greatarc/longcall/clf_by_file.log)

Interpretations:
- The statistics look much better, surprisingly good results on those minority classes.
- Perhaps the 1-second segmentation no longer works well here, considering some pulse types are very short if terms of time duration, such as `Grumble sub-pulse`.

## [18.02.2022] Experiments on Full OliveColobusDatabase Dataset

I did many experiments on classifying whether each 1-second segment from the .wav files contain a monkey call. The data source is the 284 annotated files (.wav + .txt). All of the models perform similarly and suffer from a same issue: they cannot generalize well across different .wav files.

The best f1 score on test set is 0.17 while the f1 score on training set can easily reach 0.8. However, if I mix and shuffle segments from different .wav files then split them, the f1 score on test set can also reach 0.8. 

I think either:

- the data itself does not generalize across files -> I'd be happy to try a different dataset, perhaps the one from Adriano? I haven't got that recordings and text annotations yet.
- or the training is overfit to the training set -> I actually have used some tricks to prevent overfitting (see below), perhaps ask a machine learning / audio processing expert for help?

I am more inclined to the first conclusion.

The general approach is similar, as introduced in https://github.com/bambooforest/audio_exploration/pull/1:

1. Split each .wav file to 1-second segments, extract the features for each 1-second segment, which can be the raw waveform, spectrogram, MFCC or feature vector from a pretrained model.
2. Read text annotations and match each segment (and the features) to a 0/1 class.
3. Store and split the data for training, validation, and testing. Usually, 80% of the .wav files are for training, then predict the rest 20%.
4. Train a binary classifier network. 
5. Evaluate by precision, recall, and f1 score since the classes are highly imbalanced.

### Approach 1: Spectrogram + Feed-forward Network

Extend the naive approach https://github.com/bambooforest/audio_exploration/pull/1 to the full dataset, and split data by files:

1. Segment and generate spectrogram data by `scipy`: `python data.py` -> writes data to `./data_full_clf_1/`.
2. Train and test a feed-forward network model by `scikit-learn` on CPU: `python clf.py` -> writes results to https://github.com/bambooforest/audio_exploration/blob/torch_cnn/clf.log, https://github.com/bambooforest/audio_exploration/blob/torch_cnn/clf_train.png, and https://github.com/bambooforest/audio_exploration/blob/torch_cnn/clf_test.png.

How to prevent overfitting? 

- Set `alpha=1` on `MLPClassifier`.

### Approach 2: Raw Waveform + Very Deep CNN

Approach as introduced in:
- https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
- https://arxiv.org/pdf/1610.00087.pdf

Steps:

1. Segment and generate waveform data by `torchaudio`: `python torch_data_waveform.py` -> writes data to `./data_waveform_1/` and `./data_waveform_1_split`.
2. Train and test a CNN model by `torch` on CPU/GPU: `python torch_cnn.py` -> writes results to https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_cnn.log and https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_cnn_test.png, save model to https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_cnn.pt.

How to prevent overfitting? 

- `ReduceLROnPlateau` on validation f1 score.

### Approach 3: Pretrained Wav2Vec2 Feature Vector + RNN (LSTM)

Approach as introduced in:
- https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
- https://arxiv.org/pdf/2006.11477.pdf

Steps:

1. Segment and generate Wav2Vec2 feature vector data by `torchaudio`: `python torch_data_wav2vec2.py` -> writes data to `./data_wav2vec2_1/` and `./data_wav2vec2_1_split`.
2. Train and test an RNN model by `torch` on CPU/GPU: `python torch_rnn.py` -> writes results to https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_rnn.log and https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_rnn_test.png, save model to https://github.com/bambooforest/audio_exploration/blob/torch_cnn/torch_rnn.pt.

How to prevent overfitting? 

- `ReduceLROnPlateau` on validation f1 score.
- Dropout on `nn.LSTM`.

### Further Note

- From all the material that I have read/researched, the promising way to solve this task is the combination of audio feature + neural network. The three approaches I tried performed surprisingly similarly, so I tend to believe that it is truly what we can get on this dataset by this way. Intuitively, I do not think it is such a difficult task, if people have already used similar approaches to solve tasks like speech command classification and speech recognition.
- Perhaps some insights about the dataset is needed? Like, what's the difference across the .wav files? Are they recorded and annotated by the same author? Under the same condition? Maybe too much noise from the environment? I have also heard many bird calls from the recording, do they cause a problem? ...  

## [08.12.2022] Naive Approach (on 3 files)

### To Run:

1. Generate training data: `python data.py`
2. Train and evaluate the classifier: `python clf.py` 

### Basic workflow:

1. Read the wav files and calculate spectrogram features. 

    A very important parameter here is `T_real`, how much time (second) a segment is. It determines the number of training samples, and has obvious impact on model training and performance. 
    
    I am not sure what value is most appropriate here, I guess it should be a unit that is just long enough to catch a shortest "monkey call" if existing, so I currently set it to 0.1s.

2. Read the annotated txt files and match the segments in spectrogram to the annotations, each segment gets either 0 (not a monkey call) or 1 (is a monkey call).

    The current strategy is to check whether a segment time (the middle point of the segment time space) is within any annotated range.

3. Store everything to .csv files for training. Each row is a segment, the first column is 0/1 the target, the rest columns are the "frequency bins" spectrogram features.

    I believe we can further extend it to a multi-class multi-target classification problem, adding targets such as unit, family ,etc. (base on your requirements).

4. Train a classifier. I am now using a 0.1 test ratio, probably better to use the first file and second file to predict the third file (do not do it because of limited training data). 

5. Evaluation. As the classes are highly imbalanced (hudreds of "monkey call" segments among a total of a few thousands), look at the precision and recall scores, also plot the precision-recall curve.

    https://github.com/bambooforest/audio_exploration/blob/clf/clf.log

    training:

    ![image](https://user-images.githubusercontent.com/2316987/144643050-fac284cc-da9f-490e-9116-7350924016ab.png)

    testing:

    ![image](https://user-images.githubusercontent.com/2316987/144643190-ea73f66f-d2b0-4a26-bf84-8a02ba52e0cd.png)

    The curve looks not bad at first glance, we can get a classifier with around 0.5 recall and 0.5 precision (considering we only have 3 wav files). But it also clearly shows an overfitting problem. I think we can either solve it by more training data or a much more sophisticated training process.

### Reference

- https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

