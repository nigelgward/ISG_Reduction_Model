# ISG Reduction Model
This code is useful for estimating the level of phonetic reduction in speech data.  It was developed by Javier Vazquez and Nigel Ward of the Interactive Systems Group (ISG) of the University of Texas at El Paso (UTEP), in 2024-2025.  Publications describing the data, methods, and intended uses are available via https://www.cs.utep.edu/nigel/reduction/

##Overview of the Process

The downstream model (decision head) here is simple, as most of the work is done by the HuBert features.  Thus, to predict the amount of reduction in an audio, it first needs to be converted into HuBERT features. This code provides easy access to the HuBERT model available in PyTorch. In particular, it uses the HuBERT Base model which extracts 12 layers of 768 features for every 20 ms of audio. The last layer is the one used here as it had the best performance in pilot tests. Depending on whether the audio is mono or stereo, the HuBERT model will produce one or two channels, respectively.

To used this code, fork the repository and then follow the example usage below.  Alternatively, see baby-README.md for an easier, step-by-step description

Example Usage (after forking the repository):


1. To extract the HuBERT features:

`
reduction = Reduction()
`

`
hubert_features = reduction.extract(['audio1.wav','audio2.wav','audio3.wav'])
`

Once you have the HuBERT features available, you can choose to train the model on any subset of these features for which there are corresponding labels. The labels format is a tab-delimited file in the order of Channel, Start Time, End Time, and Reduction Value. The Channel specifies if Left or Right for Stereo audio and None for Mono audio. The Start and End Time specify the timeframe for utterances specified in seconds. The Reduction value specifies the value given to the utterance of the associated timeframe. There are examples of labels under the default_data folder. If no external audio will be utilized to train, then you can skip to step2b.

2a. To train a reduction model:


`
reduction.fit(X=hubert_features,y=['labels.txt'])
`

2b. Alternatively, you can train the model on the recordings and labels
created at ISG, which are available at
www.cs.utep.edu/nigel/reduction/ .


`
reduction.default_fit()
`


Now that you have a trained model, you can use it to predict the reduction found at each frame or at each utterance from the features extracted previously. To predict reduction at each frame, we pass the features extracted previously for a single audio. The model will generate predicted reduction values for each track of the audio. In addition to frame-by-frame predictions, region-based (word-based) predictions can be generated using predict_utterance.  For this, a tab-delimited text file needs to be provided in the order of Channel, Start Time, and End Time. The format and values of the columns follow the format mentioned earlier.

3a. To predict the reduction at each 20 ms frame:

`
frame_predictions = reduction.predict(hubert_features[0])
`

3b. To predict the reduction at each utterance:

`
utterance_predictions = reduction.predict_utterances(hubert_features,='utterance_timeframes.txt')
`

If there are any questions or concerns, feel free to reach to us.
