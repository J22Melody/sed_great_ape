# audio_exploration

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

