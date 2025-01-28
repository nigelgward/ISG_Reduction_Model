# ISG Reduction Model
The Reduction Model developed at ISG available for public use. To use the model, fork the repository and follow the example usage explained below.

Example Usage:

To predict the amount of reduction in an audio, it first needs to be converted into HuBERT features. The repository provides easy access to the HuBERT model available in PyTorch. The repository uses the HuBERT Base which extracts 12 layers of 768 features for every 20 ms of audio. The last layer is the one utilized as it has the best performance during our testing. Additionally, depending on if the audio is mono or stereo, the HuBERT model will produce one or two channels, respectively.

1. To extract the HuBERT features:

`
reduction = Reduction()
`

`
hubert_features = reduction.extract(['audio1.wav','audio2.wav','audio3.wav'])
`

Once we have the HuBERT features available, you can choose to train the model on a subset of these features if there is the corresponding labels for the audio used. The labels format is a tab-delimited file in the order of Channel, Start Time, End Time, and Reduction Value. The Channel specifies if Left or Right for Stereo audio and None for Mono audio. The Start and End Time specify the timeframe for utterances specified in seconds. The Reduction value specifies the value given to the utterance of the associated timeframe. There are examples of labels under the default_data folder. If no external audio will be utilized to train, then you can skip to step2b.

2a. To train the reduction model:


`
reduction.fit(X=hubert_features,y=['labels.txt'])
`


We provide a default Reduction model that trains based off the recordings and labels completed at ISG.

2b. To train the default reduction model:


`
reduction.default_fit()
`


Now that we have a trained model, we can utilize it to predict the reduction found at each frame or at each utterance from the features extracted previously. To predict reduction at each frame, we pass the features extracted previously for a single audio. The model will generate reduction values for each track of the audio. To predict reduction at each utterance, a tab-delimited text file needs to be provided in the order of Channel, Start Time, and End Time. The format and values of the columns follow the trends previously established in Step 2.

3a. To predict the reduction at each 20 ms frame:

`
frame_predictions = reduction.predict(X=hubert_features)
`

3b. To predict the reduction at each utterance:

`
utterance_predictions = reduction.predict_utterance(X=hubert_features,timeframes='utterance_timeframes.txt')
`

If there are any questions or concerns, feel free to reach out at: 
