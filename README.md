# audio_exploration

## To Run:

1. Generate training data: `python data.py`
2. Train and evaluate the classifier: `python clf.py` 

## Basic workflow:

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

## Reference

- https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

